import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict

# 抑制 scapy 的 TLS / cryptography WARNING（不影响解析结果，仅打印噪音）
logging.getLogger("scapy").setLevel(logging.ERROR)

import h5py
import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, Raw
from scapy.layers.tls.handshake import TLSClientHello
from scapy.layers.tls.record import TLS

# ==================== 常量（论文 §3.2 §3.3） ====================
MAX_PKT_LEN = 200       # P 分支最大包数
MAX_BURST_LEN = 16      # B 分支最大突发数
BURST_FEAT_DIM = 4      # B 每突发动特征: cnt, tot_bytes, dur, avg_len
S_DIM = 32              # S 分支维度
JA3_BUCKET_BITS = 6     # JA3 hash 桶位数 (2^6=64 桶)
SUITE_EMB_DIM = 32      # 套件 embedding 维度
EXT_EMB_DIM = 32        # 扩展 embedding 维度
H_DIM = JA3_BUCKET_BITS + SUITE_EMB_DIM + EXT_EMB_DIM + 1  # 71

# ==================== 模块 1: PCAP 流聚合 ====================
class PcapFlowAggregator:
    """scapy 读 PCAP，按双向 5 元组聚合流

    每条流收集：
      - ts_list: 包时间戳 (秒，相对 PCAP 起始)
      - len_list: 包长度（带方向：正向+，反向-）
      - ja3: ClientHello 解析结果（json 字符串或 None）
      - flow_key: 5 元组规范化字符串
    """
    def __init__(self):
        self.flows = {}   # flow_key → dict(ts, len, ja3)

    @staticmethod
    def norm_key(sip, sport, dip, dport, proto):
        """规范化 5 元组键：双向共享同一 key"""
        a = (sip, int(sport)) if sport is not None else (sip, 0)
        b = (dip, int(dport)) if dport is not None else (dip, 0)
        if a <= b:
            return f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}"
        return f"{b[0]}:{b[1]}-{a[0]}:{a[1]}-{proto}"

    @staticmethod
    def ja3_from_hello(hello):
        """TLSClientHello → JA3 字符串

        JA3 = TLSVersion,Ciphers,Extensions,EllipticCurves,ECPointFormats
        注：scapy 中部分 TLS_Ext 的 __str__ 返回 bytes（TLS_Ext_Unknown 等），
        这里用安全转换器避免 join 抛异常。
        """
        def safe(x):
            """把任意对象转字符串，避免 bytes 触发 join 异常"""
            try:
                s = str(x)
                if isinstance(s, bytes):
                    s = s.decode("utf-8", errors="replace")
                return s
            except Exception:
                try:
                    return repr(x)
                except Exception:
                    return ""
        try:
            v = hello.version
            ciphers = "-".join(safe(x) for x in (hello.ciphers or []))
            exts = "-".join(safe(x) for x in (hello.ext or []))
            # 椭圆曲线与点格式在 ext 内（ec_curves, ec_point），需要从原始 hello 提取
            ec_curves = ""
            ec_points = ""
            if hasattr(hello, "ext"):
                for e in (hello.ext or []):
                    if hasattr(e, "curves") and e.curves:
                        ec_curves = "-".join(safe(x) for x in e.curves)
                    if hasattr(e, "formats") and e.formats:
                        ec_points = "-".join(safe(x) for x in e.formats)
            return f"{v},{ciphers},{exts},{ec_curves},{ec_points}"
        except Exception:
            return None

    def feed_pcap(self, pcap_path, label_hint=None, max_packets=None):
        """读单个 PCAP，返回 (flow_dict, label_hint)

        label_hint 来自目录名/文件名（如 BitTorrent → "BitTorrent"）。
        """
        flows_local = {}
        ts0 = None
        n = 0
        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    if max_packets and n >= max_packets:
                        break
                    n += 1
                    if not pkt.haslayer(IP):
                        continue
                    ip = pkt[IP]
                    proto = 6 if pkt.haslayer(TCP) else (17 if pkt.haslayer(UDP) else 0)
                    if proto == 0:
                        continue
                    sport = pkt[TCP].sport if pkt.haslayer(TCP) else pkt[UDP].sport
                    dport = pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport
                    key = self.norm_key(ip.src, sport, ip.dst, dport, proto)

                    ts = float(pkt.time)
                    if ts0 is None:
                        ts0 = ts
                    # 包长：正向为 +，反向为 -
                    if (ip.src, sport) <= (ip.dst, dport):
                        signed_len = +len(pkt)
                    else:
                        signed_len = -len(pkt)

                    if key not in flows_local:
                        flows_local[key] = {
                            "ts": [], "len": [], "ja3": None, "hint": label_hint,
                        }
                    rec = flows_local[key]
                    rec["ts"].append(ts - ts0)
                    rec["len"].append(signed_len)

                    # JA3 提取：ClientHello
                    if pkt.haslayer(TLS) and rec["ja3"] is None:
                        tls = pkt[TLS]
                        if hasattr(tls, "msg") and tls.msg and isinstance(tls.msg[0], TLSClientHello):
                            rec["ja3"] = self.ja3_from_hello(tls.msg[0])
        except Exception as e:
            print(f"  ! pcap read error ({pcap_path}): {e}", file=sys.stderr)
        return flows_local

    def feed_pcap_dir(self, pcap_dir, exts=(".pcap", ".pcapng"),
                      label_from=None, max_per_file=None, max_pcap_per_dir=None):
        """遍历目录下所有 PCAP，合并到 self.flows

        label_from: "filename" / "parent" / "second_parent" / None
        顶层 pcap（parent 在 skip_dirs）时回退到 filename。

        max_pcap_per_dir: 每个二级目录最多取的 pcap 数（DoHBrw 采样用）
        """
        skip_dirs = {"pcap", "pcaps", "PCAP", "PCAPs"}
        # 按 second_parent 分组（DoHBrw 用 second_parent 作 label）
        from collections import defaultdict
        by_group = defaultdict(list)
        for root, _, files in os.walk(pcap_dir):
            for f in files:
                if f.endswith(exts):
                    full = os.path.join(root, f)
                    if label_from == "second_parent":
                        # group by second_parent (e.g. BenignDoH-Chrome-AdGuard)
                        try:
                            second_parent = os.path.basename(
                                os.path.dirname(os.path.dirname(full)))
                        except Exception:
                            second_parent = "default"
                    else:
                        second_parent = "all"
                    by_group[second_parent].append(full)
        # 采样
        paths = []
        for grp, ps in sorted(by_group.items()):
            ps.sort()
            if max_pcap_per_dir and len(ps) > max_pcap_per_dir:
                # 等距采样
                step = max(1, len(ps) // max_pcap_per_dir)
                ps = ps[::step][:max_pcap_per_dir]
            paths.extend(ps)
        paths.sort()
        print(f"  [scan] {len(paths)} pcap files under {pcap_dir} "
              f"(max_pcap_per_dir={max_pcap_per_dir})")
        for p in paths:
            parent = os.path.basename(os.path.dirname(p))
            if label_from == "filename":
                hint = os.path.splitext(os.path.basename(p))[0]
            elif label_from == "parent":
                # 顶层 pcap：parent 是 pcap/ 时回退到 filename
                if parent in skip_dirs:
                    hint = os.path.splitext(os.path.basename(p))[0]
                else:
                    hint = parent
            elif label_from == "second_parent":
                # DoHBrw 目录深度不统一：BenignDoH 是 PCAP/<group>/<resolver>/<file>，
                # MaliciousDoH 是 PCAP/<group>/<file>。second_parent 只对 Benign 有效。
                # 这里改用「向上找第一个 BenignDoH*/MaliciousDoH* 祖先」，两种深度都覆盖。
                hint = None
                parts = p.replace("\\", "/").split("/")
                for i in range(len(parts) - 1, -1, -1):
                    name = parts[i]
                    if name.startswith("BenignDoH") or name.startswith("MaliciousDoH"):
                        hint = name
                        break
                if hint is None:
                    hint = os.path.basename(os.path.dirname(p))
            else:
                hint = None
            sub = self.feed_pcap(p, label_hint=hint, max_packets=max_per_file)
            for k, v in sub.items():
                if k not in self.flows:
                    self.flows[k] = v
                else:
                    # 合并（罕见，跨文件同流）
                    self.flows[k]["ts"].extend(v["ts"])
                    self.flows[k]["len"].extend(v["len"])
                    if v["ja3"] and not self.flows[k]["ja3"]:
                        self.flows[k]["ja3"] = v["ja3"]
            print(f"    [{len(paths)}] {os.path.basename(p)} (parent={parent}): "
                  f"flows={len(sub)} cum={len(self.flows)} hint={hint}")
        return self.flows

# ==================== 模块 2: 4 分支特征计算 ====================
class FeatureComputer:
    """从 PcapFlowAggregator 输出计算 P/B/S/H 4 分支特征"""

    @staticmethod
    def _seq_to_burst(pkt_signed_lens, max_burst=MAX_BURST_LEN):
        """包序列 → 突发序列 [cnt, tot_bytes, dur, avg_len]"""
        bursts = []
        if not pkt_signed_lens:
            return np.zeros((max_burst, BURST_FEAT_DIM), dtype=np.float32)
        cur_dir = 1 if pkt_signed_lens[0] >= 0 else -1
        cnt, tot = 1, abs(pkt_signed_lens[0])
        t0, t1 = 0.0, 1.0   # 占位 t0/t1（无 IAT 信息，用包序号近似 dur）
        for s in pkt_signed_lens[1:]:
            d = 1 if s >= 0 else -1
            if d == cur_dir:
                cnt += 1; tot += abs(s); t1 += 1.0
            else:
                bursts.append([cnt, tot, max(t1 - t0, 1e-4), tot / max(cnt, 1)])
                cur_dir = d; cnt = 1; tot = abs(s); t0, t1 = t1, t1 + 1.0
        bursts.append([cnt, tot, max(t1 - t0, 1e-4), tot / max(cnt, 1)])
        bf = np.array(bursts[:max_burst], dtype=np.float32)
        if len(bf) < max_burst:
            pad = np.zeros((max_burst - len(bf), BURST_FEAT_DIM), dtype=np.float32)
            bf = np.vstack([bf, pad])
        return bf

    @staticmethod
    def compute_p(flow):
        """P 分支：(2, MAX_PKT_LEN) = [包长, IAT]"""
        L = flow["len"]
        T = flow["ts"]
        n = min(len(L), MAX_PKT_LEN)
        if n == 0:
            return np.zeros((2, MAX_PKT_LEN), dtype=np.float32)
        pkt_len = np.zeros(MAX_PKT_LEN, dtype=np.float32)
        iat = np.zeros(MAX_PKT_LEN, dtype=np.float32)
        pkt_len[:n] = np.abs(L[:n])
        if n > 1:
            dt = np.diff(T[:n]).astype(np.float32) * 1000.0   # s→ms
            iat[1:n] = dt
        return np.stack([pkt_len, iat], axis=0)

    @staticmethod
    def compute_b(flow):
        """B 分支：(MAX_BURST_LEN, BURST_FEAT_DIM)"""
        return FeatureComputer._seq_to_burst(flow["len"])

    @staticmethod
    def compute_s(flow):
        """S 分支：32 维流级统计（论文 §3.2）

        注意：此处是从 PCAP 包序列计算的近似版本（CICFlowMeter 风格）。
        真实 CICFlowMeter 会另算活跃/空闲时长等。
        """
        L = np.array(flow["len"], dtype=np.float64) if flow["len"] else np.array([0.0])
        abs_L = np.abs(L)
        n = len(abs_L)
        if n == 0:
            return np.zeros(S_DIM, dtype=np.float32)

        # 正向（src2dst）/反向（dst2src）拆分
        pos = abs_L[L >= 0]
        neg = abs_L[L < 0]
        np_, nn_ = len(pos), len(neg)

        # IAT（按时间戳）
        T = flow["ts"]
        iat_ms = np.diff(T).astype(np.float64) * 1000.0 if n > 1 else np.array([0.0])

        # TCP 标志位（暂设 0，PCAP 解析可补）
        syn = ack = psh = fin = urg = cwr = 0.0

        feats = np.zeros(S_DIM, dtype=np.float32)
        # 0: bidirectional_packets
        feats[0] = n
        # 1: bidirectional_bytes
        feats[1] = abs_L.sum()
        # 2: src2dst_packets, 3: src2dst_bytes
        feats[2] = np_; feats[3] = pos.sum() if np_ > 0 else 0.0
        # 4-7: src2dst 包长/IAT 聚合
        if np_ > 0:
            feats[4] = pos.mean(); feats[5] = pos.std() if np_ > 1 else 0.0
            feats[6] = 0.0; feats[7] = 0.0   # src2dst IAT 需要按时戳拆，此处省略
            feats[8] = pos.max(); feats[9] = pos.min()
        # 10: dst2src_packets, 11: dst2src_bytes
        feats[10] = nn_; feats[11] = neg.sum() if nn_ > 0 else 0.0
        # 12-15: dst2src 包长/IAT 聚合
        if nn_ > 0:
            feats[12] = neg.mean(); feats[13] = neg.std() if nn_ > 1 else 0.0
            feats[14] = 0.0; feats[15] = 0.0
            feats[16] = neg.max(); feats[17] = neg.min()
        # 18-21: bidirectional 包长聚合
        feats[18] = abs_L.mean(); feats[19] = abs_L.std() if n > 1 else 0.0
        feats[20] = abs_L.min(); feats[21] = abs_L.max()
        # 22-25: bidirectional IAT 聚合
        if len(iat_ms) > 0:
            feats[22] = iat_ms.mean(); feats[23] = iat_ms.std() if len(iat_ms) > 1 else 0.0
            feats[24] = iat_ms.min() if len(iat_ms) > 0 else 0.0
            feats[25] = iat_ms.max() if len(iat_ms) > 0 else 0.0
        # 26-31: TCP flags + 备用
        feats[26] = syn; feats[27] = ack; feats[28] = psh
        feats[29] = fin; feats[30] = urg; feats[31] = cwr
        return feats

    @staticmethod
    def _ja3_buckets(ja3_str):
        """JA3 字符串 → 6 维 one-hot（md5 hash → 桶，6 个最常见桶）"""
        if ja3_str is None:
            return np.zeros(JA3_BUCKET_BITS, dtype=np.float32)
        h = hashlib.md5(str(ja3_str).encode("utf-8", errors="ignore")).digest()
        bucket = h[0] % JA3_BUCKET_BITS   # 0..5
        v = np.zeros(JA3_BUCKET_BITS, dtype=np.float32)
        v[bucket] = 1.0
        return v

    @staticmethod
    def _id_seq_embedding(ja3_str, dim, sep="-"):
        """JA3 id 序列 → 确定性 embedding（hash → [-1, 1]）"""
        if ja3_str is None:
            return np.zeros(dim, dtype=np.float32)
        parts = str(ja3_str).split(",")
        if len(parts) < 3:
            return np.zeros(dim, dtype=np.float32)
        seq_str = parts[1] if dim == SUITE_EMB_DIM else (
            parts[2] if len(parts) > 2 else "")
        if not seq_str:
            return np.zeros(dim, dtype=np.float32)
        emb = np.zeros(dim, dtype=np.float32)
        ids = [x for x in seq_str.split(sep) if x]
        for i, x in enumerate(ids[:8]):
            h = hashlib.md5(x.encode("utf-8", errors="ignore")).digest()
            emb[i * 4 % dim] = (h[0] / 255.0 - 0.5) * 2
        return emb

    @classmethod
    def compute_h(cls, flow):
        """H 分支：71 维握手层"""
        ja3 = flow.get("ja3")
        bucket = cls._ja3_buckets(ja3)
        suite = cls._id_seq_embedding(ja3, SUITE_EMB_DIM)
        ext = cls._id_seq_embedding(ja3, EXT_EMB_DIM)
        miss = np.array([0.0 if ja3 else 1.0], dtype=np.float32)
        return np.concatenate([bucket, suite, ext, miss]).astype(np.float32)

# ==================== 模块 3: 标签 join ====================
class LabelJoiner:
    """根据数据集类型从 CSV 或目录结构推断标签"""

    @staticmethod
    def join_cic(csv_dir, flows):
        """CIC-IDS2017: 用 csv_with_timestamp/ 目录的 5 元组 + Proto + Timestamp join

        CSV (2.4M 行) 含 src_ip/dst_ip/src_port/dst_port/protocol/timestamp/Label
        与 PCAP 流的 5 元组做 join。
        """
        import glob
        csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        print(f"  [join_cic] loading {len(csv_paths)} CSVs from {csv_dir}")
        dfs = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p, low_memory=False)
                df.columns = [c.strip() for c in df.columns]
                dfs.append(df)
            except UnicodeDecodeError:
                # CIC-IDS2017 中部分 CSV 含 0x96 字节，utf-8 解不开；
                # 回退 latin-1 容错以保留全部行
                try:
                    df = pd.read_csv(p, low_memory=False, encoding="latin-1",
                                     encoding_errors="replace")
                    df.columns = [c.strip() for c in df.columns]
                    dfs.append(df)
                except Exception as e:
                    print(f"    ! {p}: {e}")
            except Exception as e:
                print(f"    ! {p}: {e}")
        df = pd.concat(dfs, ignore_index=True)
        print(f"  [join_cic] total CSV rows: {len(df)}")

        # 构造归一化 5 元组键（与 PcapFlowAggregator.norm_key 一致）
        def row_key(row):
            try:
                # GeneratedLabelledFlows 列名
                sip = str(row.get("Source IP", row.get("src_ip", ""))).strip()
                sport = int(row.get("Source Port", row.get("src_port", 0)))
                dip = str(row.get("Destination IP", row.get("dst_ip", ""))).strip()
                dport = int(row.get("Destination Port", row.get("dst_port", 0)))
                proto = int(row.get("Protocol", row.get("protocol", 6)))
            except Exception:
                return None
            if not sip or not dip:
                return None
            return PcapFlowAggregator.norm_key(sip, sport, dip, dport, proto)

        # 构造 5 元组键并去重
        df["_key"] = df.apply(row_key, axis=1)
        df = df.dropna(subset=["_key"])
        if "Label" not in df.columns and "label" in df.columns:
            df["Label"] = df["label"]
        df = df.dropna(subset=["Label"])
        # 为节省内存，按流键去重（同一 5 元组可能多次出现）
        # 策略：优先攻击类（非 BENIGN）→ BENIGN 兜底。
        # 原因：CIC-IDS2017 中攻击会话通常伴随大量同 5 元组的正常/工具流量，
        # 若用 mode，BENIGN 会淹没真实攻击 label；先取攻击类（语义正确性 > 多数类）。
        from collections import Counter
        label_map = {}
        for key, group in df.groupby("_key"):
            cnt = Counter(group["Label"])
            non_benign = {l: c for l, c in cnt.items() if l != "BENIGN"}
            if non_benign:
                label_map[key] = max(non_benign, key=non_benign.get)
            else:
                label_map[key] = "BENIGN"
        print(f"  [join_cic] label_map entries: {len(label_map)} "
              f"(multi-label groups: "
              f"{sum(1 for k, g in df.groupby('_key') if g['Label'].nunique() > 1)})")

        # 应用到 flows
        n_hit = 0
        for k, v in flows.items():
            if k in label_map:
                v["label"] = label_map[k]
                n_hit += 1
            else:
                v["label"] = "UNKNOWN"
        return n_hit, len(flows)

    @staticmethod
    def join_ustc(pcap_label_map, flows):
        """USTC: 标签从父目录名（如 BitTorrent/）

        pcap_label_map: {pcap_filename → label_name}
        """
        # 在 feeds 时已写入 hint，这里把 hint 提升为 label
        # 跳过目录元名（pcap/PCAP/PCAPS 等）
        skip_dirs = {"pcap", "pcaps", "PCAP", "PCAPs"}
        n_hit = 0
        for k, v in flows.items():
            hint = v.get("hint")
            if hint and hint not in skip_dirs:
                v["label"] = hint
                n_hit += 1
            else:
                v["label"] = "UNKNOWN"
        return n_hit, len(flows)

    @staticmethod
    def join_iscx(pcap_label_map, flows):
        """ISCX: 标签从祖父目录（NonVPN-* vs VPN-*）"""
        n_hit = 0
        for k, v in flows.items():
            if v.get("hint"):
                # hint 是父目录名（NonVPN-PCAPs-01, VPN-PCAPS-01, ...）
                if "NonVPN" in v["hint"]:
                    v["label"] = "NonVPN"
                elif "VPN" in v["hint"]:
                    v["label"] = "VPN"
                else:
                    v["label"] = "UNKNOWN"
                n_hit += 1
            else:
                v["label"] = "UNKNOWN"
        return n_hit, len(flows)

    @staticmethod
    def join_dohbrw(pcap_label_map, flows):
        """DoHBrw: 标签从 second_parent（BenignDoH-* / MaliciousDoH-*）

        hint 可能是 "BenignDoH_NonDoH-Chrome-AdGuard" / "MaliciousDoH-dns2tcp-Pcap-001_600" / "PCAP"（3 层路径）
        含关键词即推断为 Benign / Malicious。
        """
        n_hit = 0
        for k, v in flows.items():
            hint = v.get("hint")
            if hint and "Benign" in hint:
                v["label"] = "Benign"
                n_hit += 1
            elif hint and "Malicious" in hint:
                v["label"] = "Malicious"
                n_hit += 1
            else:
                v["label"] = "UNKNOWN"
        return n_hit, len(flows)

# ==================== 模块 4: h5 写入器 ====================
class H5Writer:
    """统一 h5 格式：(N, 2, 200) P / (N, 16, 4) B / (N, 32) S / (N, 71) H / labels"""
    def __init__(self, out_path, chunk_size=10000):
        self.out_path = out_path
        self.chunk = chunk_size
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self._init()

    def _init(self):
        with h5py.File(self.out_path, "w") as h:
            h.create_dataset("pkt_seq", shape=(0, 2, MAX_PKT_LEN),
                             maxshape=(None, 2, MAX_PKT_LEN),
                             dtype="float32", chunks=(self.chunk, 2, MAX_PKT_LEN))
            h.create_dataset("burst_seq", shape=(0, MAX_BURST_LEN, BURST_FEAT_DIM),
                             maxshape=(None, MAX_BURST_LEN, BURST_FEAT_DIM),
                             dtype="float32", chunks=(self.chunk, MAX_BURST_LEN, BURST_FEAT_DIM))
            h.create_dataset("stat", shape=(0, S_DIM),
                             maxshape=(None, S_DIM), dtype="float32",
                             chunks=(self.chunk, S_DIM))
            h.create_dataset("hand", shape=(0, H_DIM),
                             maxshape=(None, H_DIM), dtype="float32",
                             chunks=(self.chunk, H_DIM))
            h.create_dataset("label", shape=(0,), maxshape=(None,),
                             dtype=h5py.string_dtype(encoding="utf-8"),
                             chunks=(self.chunk,))
            h.create_dataset("flow_key", shape=(0,), maxshape=(None,),
                             dtype=h5py.string_dtype(encoding="utf-8"),
                             chunks=(self.chunk,))
            h.attrs["stat_feats"] = [
                "bidirectional_packets", "bidirectional_bytes",
                "src2dst_packets", "src2dst_bytes",
                "src2dst_mean_ps", "src2dst_stddev_ps",
                "src2dst_mean_piat_ms", "src2dst_stddev_piat_ms",
                "src2dst_max_ps", "src2dst_min_ps",
                "dst2src_packets", "dst2src_bytes",
                "dst2src_mean_ps", "dst2src_stddev_ps",
                "dst2src_mean_piat_ms", "dst2src_stddev_piat_ms",
                "dst2src_max_ps", "dst2src_min_ps",
                "bidirectional_mean_ps", "bidirectional_stddev_ps",
                "bidirectional_min_ps", "bidirectional_max_ps",
                "bidirectional_mean_piat_ms", "bidirectional_stddev_piat_ms",
                "bidirectional_min_piat_ms", "bidirectional_max_piat_ms",
                "bidirectional_syn_packets", "bidirectional_ack_packets",
                "bidirectional_psh_packets", "bidirectional_fin_packets",
                "bidirectional_urg_packets", "bidirectional_cwr_packets",
            ]

    def append_batch(self, pkt_list, burst_list, stat_list, hand_list,
                     label_list, key_list):
        with h5py.File(self.out_path, "a") as h:
            n = len(label_list)
            old = h["pkt_seq"].shape[0]
            new = old + n
            h["pkt_seq"].resize((new, 2, MAX_PKT_LEN))
            h["burst_seq"].resize((new, MAX_BURST_LEN, BURST_FEAT_DIM))
            h["stat"].resize((new, S_DIM))
            h["hand"].resize((new, H_DIM))
            h["label"].resize((new,))
            h["flow_key"].resize((new,))
            h["pkt_seq"][old:new] = np.stack(pkt_list)
            h["burst_seq"][old:new] = np.stack(burst_list)
            h["stat"][old:new] = np.stack(stat_list)
            h["hand"][old:new] = np.stack(hand_list)
            h["label"][old:new] = np.array(label_list, dtype=h5py.string_dtype())
            h["flow_key"][old:new] = np.array(key_list, dtype=h5py.string_dtype())

# ==================== 模块 5: 主流程 ====================
def process_dataset(name, cfg, out_path, max_packets_per_file=None,
                    max_pcap_per_dir=None):
    """单个数据集的完整预处理流程"""
    print(f"\n=== {name} ===")
    t0 = time.time()

    # 1. PCAP 聚合
    aggregator = PcapFlowAggregator()
    aggregator.feed_pcap_dir(
        cfg["pcap_dir"], label_from=cfg.get("label_from", "parent"),
        max_per_file=max_packets_per_file,
        max_pcap_per_dir=max_pcap_per_dir)
    flows = aggregator.flows
    print(f"  [agg] total flows = {len(flows)}")

    # 2. 标签 join
    if "csv" in cfg:
        joiner = getattr(LabelJoiner, f"join_{cfg['joiner']}")
        n_hit, n_total = joiner(cfg["csv"], flows)
        print(f"  [label] {n_hit}/{n_total} flows labeled")
    else:
        n_hit, n_total = LabelJoiner.__dict__[f"join_{cfg['joiner']}"](None, flows)
        print(f"  [label] {n_hit}/{n_total} flows labeled")

    # 过滤掉 UNKNOWN
    flows = {k: v for k, v in flows.items() if v.get("label") and v["label"] != "UNKNOWN"}
    print(f"  [filter] after UNKNOWN remove = {len(flows)}")

    # 3. 4 分支特征计算
    writer = H5Writer(out_path)
    BATCH = 5000
    pkt_buf, burst_buf, stat_buf, hand_buf = [], [], [], []
    label_buf, key_buf = [], []
    n_written = 0
    for k, v in flows.items():
        pkt_buf.append(FeatureComputer.compute_p(v))
        burst_buf.append(FeatureComputer.compute_b(v))
        stat_buf.append(FeatureComputer.compute_s(v))
        hand_buf.append(FeatureComputer.compute_h(v))
        label_buf.append(v["label"])
        key_buf.append(k)
        if len(label_buf) >= BATCH:
            writer.append_batch(pkt_buf, burst_buf, stat_buf, hand_buf,
                                label_buf, key_buf)
            n_written += len(label_buf)
            pkt_buf, burst_buf, stat_buf, hand_buf = [], [], [], []
            label_buf, key_buf = [], []
            print(f"  [write] {n_written}/{len(flows)}")
    if label_buf:
        writer.append_batch(pkt_buf, burst_buf, stat_buf, hand_buf,
                            label_buf, key_buf)
        n_written += len(label_buf)
    print(f"  [done] {n_written} flows → {out_path} ({time.time()-t0:.1f}s)")

    # 4. 标签分布
    label_counts = {}
    for v in flows.values():
        lbl = v["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"  [dist] {sorted(label_counts.items(), key=lambda x: -x[1])[:10]}")
    return label_counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["cicids2017", "ustc", "iscx", "dohbrw"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_packets_per_file", type=int, default=None,
                    help="限每个 PCAP 解析包数（烟雾测试用）")
    ap.add_argument("--max_pcap_per_dir", type=int, default=None,
                    help="每个二级目录最多取几个 pcap（DoHBrw 采样用）")
    args = ap.parse_args()

    DATASETS = {
        "cicids2017": {
            "pcap_dir": r"D:/dataset/CICDS2017/pcap",
            "csv": r"D:/dataset/CICDS2017/GeneratedLabelledFlows/TrafficLabelling",
            "joiner": "cic",
            "label_from": "parent",
        },
        "ustc": {
            "pcap_dir": r"D:/dataset/USTC-TFC2016/pcap",
            "joiner": "ustc",
            "label_from": "parent",
        },
        "iscx": {
            "pcap_dir": r"D:/dataset/ISCX-VPN/PCAP",
            "joiner": "iscx",
            "label_from": "parent",
        },
        "dohbrw": {
            "pcap_dir": r"D:/dataset/dohbrw-2020/PCAP",
            "joiner": "dohbrw",
            "label_from": "second_parent",
        },
    }
    cfg = DATASETS[args.dataset]
    process_dataset(args.dataset, cfg, args.out, args.max_packets_per_file,
                    args.max_pcap_per_dir)

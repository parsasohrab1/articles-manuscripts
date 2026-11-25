import argparse
import pathlib

from . import audit, data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qrngaudit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="audit bit strings per device and circuit")
    r.add_argument("--csv", help="CSV with device_id,circuit_id,bitstring (default: synthetic ideal / biased / correlated devices)")
    r.add_argument("--strings", type=int, default=40, help="strings per circuit for synthetic data")
    r.add_argument("--length", type=int, default=1024)
    r.add_argument("--q", type=float, default=0.05, help="target false discovery rate")
    r.add_argument("--method", default="binom", choices=["binom", "ks"])
    r.add_argument("--scope", default="device", choices=["device", "global"], help="BH correction inside each device or over the whole fleet")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    if a.csv:
        table = data.load_csv(a.csv)
    else:
        spec = {"dev_ideal_1": "ideal", "dev_ideal_2": "ideal", "dev_biased": "biased", "dev_correlated": "correlated"}
        table = data.make_table(spec, a.strings, a.length, a.seed)
    g, dev, circ = audit.audit(table, a.q, a.method, a.scope)
    far = audit.empirical_false_alarm_rate(30, a.strings, a.length, a.q, a.method, a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "tests.csv", index=False)
    dev.to_csv(out / "devices.csv", index=False)
    circ.to_csv(out / "circuits.csv", index=False)
    print(dev.round(4).to_string(index=False))
    print(f"empirical false-alarm rate on 30 ideal devices at q={a.q}: {far:.3f}")


if __name__ == "__main__":
    main()

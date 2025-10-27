import re

def is_section_header(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if re.match(r'^[+-]?\d', s):
        return False
    return bool(re.match(r'^[A-Za-z]', s))

def sort_and_trim_atoms(input_file, output_sorted, output_other,
                        sort_col=0, keep_ncols=5):
    header_lines = []
    atom_lines = []
    other_lines = []
    atoms_section = False
    took_blank_after_atoms = False
    in_other = False

    with open(input_file, "r") as f:
        for line in f:

            # まだAtoms前
            if not atoms_section:
                header_lines.append(line)
                if line.strip().startswith("Atoms"):
                    atoms_section = True
                continue

            # Atoms直後の空行保持
            if atoms_section and not took_blank_after_atoms:
                if line.strip() == "":
                    header_lines.append(line)
                    continue
                else:
                    took_blank_after_atoms = True

            # Atoms終了判定
            if is_section_header(line):
                in_other = True
                other_lines.append(line)
                continue

            # Atoms データ中
            if not in_other:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                atom_lines.append(line.rstrip("\n"))
            else:
                other_lines.append(line)

    if not atom_lines:
        raise RuntimeError("Atomsのデータ行が見つかりません")

    # ソート
    def key_fn(s):
        try:
            return int(s.split()[sort_col])
        except Exception:
            return (1 << 62)
    atom_lines.sort(key=key_fn)

    # 列トリム
    trimmed = []
    for s in atom_lines:
        parts = s.split()
        parts = parts[:keep_ncols]
        trimmed.append(" ".join(parts))

    # 書き出し: sorted data
    with open(output_sorted, "w") as f:
        f.writelines(header_lines)
        for ln in trimmed:
            f.write(ln + "\n")

    # 書き出し: other.data
    with open(output_other, "w") as f:
        f.writelines(other_lines)

    print(f"Sorted Atoms saved to: {output_sorted}")
    print(f"Removed sections saved to: {output_other}")


# 使い方例:
sort_and_trim_atoms("aSi.data", "aSi_sorted.data", "other.data",
                     sort_col=0, keep_ncols=5)

// トグルスイッチ (チェックボックスの見た目置換、prompts/70)。
// なぜ: 素のチェックボックスはダークテーマの中で浮いて見え、かつラベルとの位置関係が
// バラバラ (離れて配置されるもの/隣接するものが混在) で「どの項目のON/OFFか」が
// 目で追いにくかった。共通コンポーネント化し、label を渡すことでテキスト+スイッチが
// 8px ギャップで隣接した1つの <label> になる (テキストクリックでも切り替え可能)。
// 実体は hidden な <input type="checkbox"> のままなので、フォーカス/キーボード操作
// (Space で切替) やスクリーンリーダーの読み上げはネイティブ挙動を維持する。
export function Toggle({
  checked,
  onChange,
  label,
  disabled,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: string;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className={`toggle${disabled ? " disabled" : ""}`} title={title}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="toggle-track">
        <span className="toggle-knob" />
      </span>
      {label !== undefined && <span className="toggle-label">{label}</span>}
    </label>
  );
}

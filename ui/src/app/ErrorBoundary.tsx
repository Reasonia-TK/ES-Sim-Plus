// 描画中の例外で真っ白にならないよう、内容と再読み込みボタンを出す (v1 main.tsx と同じ)。
// 未保存の変更は自動保存の写しから次の起動時に復元できる。

import { Component, type ErrorInfo, type ReactNode } from "react";
import { writeRecovery } from "../io/autosave";

interface State {
  error: Error | null;
  stack: string;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, stack: "" };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ stack: `${error.stack ?? ""}\n${info.componentStack ?? ""}` });
    writeRecovery();
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="crash">
        <h1>ES-Sim: 画面の表示中にエラーが起きました / A rendering error occurred</h1>
        <p>{this.state.error.message}</p>
        <pre className="json">{this.state.stack}</pre>
        <button type="button" className="button primary" onClick={() => window.location.reload()}>
          再読み込み / Reload
        </button>
      </div>
    );
  }
}

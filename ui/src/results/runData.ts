// 実行の結果・入力・ライブ表示のデータを画面で使う形で取る (必要になったら取りに行き、覚えておく)。

import { useEffect, useState } from "react";
import { errorText } from "../app/messages";
import { jobProject } from "../jobs/api";
import { fetchResult, useJobs, watchJobs } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import type { Project } from "../model/project";
import { useResultsView } from "./resultsView";

/** 結果 (完了・停止した実行、同じ実行なら覚えているもの) */
export function useRunResult<T>(job: JobSummary | undefined): { result: T | null; loading: boolean; error: string | null } {
  const cached = useJobs((s) => (job ? s.results[job.id] : undefined));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fresh = cached && job && cached.run === job.runs ? (cached.data as T) : null;
  const want = Boolean(job?.has_result) && !fresh;
  const id = job?.id;
  const runs = job?.runs;
  useEffect(() => {
    if (!want || !id) return;
    let alive = true;
    setLoading(true);
    setError(null);
    fetchResult<T>(id)
      .catch((e) => alive && setError(errorText(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [want, id, runs]);
  return { result: fresh, loading: want && loading, error };
}

/** 実行の入力 (この画面から出したものは覚えている、ほかはバックエンドから取る) */
export function useRunProject(job: JobSummary | undefined): Project | null {
  const input = useJobs((s) => (job ? s.inputs[job.id] : undefined));
  const id = job?.id;
  useEffect(() => {
    if (!id || input?.project) return;
    let alive = true;
    void jobProject(id)
      .then((r) => {
        if (!alive || !r.project) return;
        useJobs.setState((s) => ({ inputs: { ...s.inputs, [id]: { project: r.project, docSerial: s.inputs[id]?.docSerial ?? -1 } } }));
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [id, input?.project]);
  return input?.project ?? null;
}

/** 最新のライブのフレーム (走っている間、見ている実行だけ届く。終わっても最後のものは残る) */
export function useFrame<T>(job: JobSummary | undefined): T | undefined {
  return useJobs((s) => (job ? s.frames[job.id] : undefined)) as unknown as T | undefined;
}

/** ビューア・グラフに出している実行 */
export function useActiveJob(): JobSummary | undefined {
  const id = useResultsView((s) => s.activeRun);
  return useJobs((s) => (id ? s.jobs[id] : undefined));
}

const watchCounts = new Map<string, number>();

/** 走っている間このジョブのライブのフレームを受ける (見ている所ごとに数え、誰も見なくなったらやめる) */
export function useWatchJob(id: string | null): void {
  useEffect(() => {
    if (!id) return;
    watchCounts.set(id, (watchCounts.get(id) ?? 0) + 1);
    watchJobs([...watchCounts.keys()]);
    return () => {
      const n = (watchCounts.get(id) ?? 1) - 1;
      if (n > 0) watchCounts.set(id, n);
      else watchCounts.delete(id);
      watchJobs([...watchCounts.keys()]);
    };
  }, [id]);
}

/** 出している実行が走っている間はライブのフレームを受ける (アプリに 1 つ) */
export function useWatchActiveRun(): void {
  const job = useActiveJob();
  useWatchJob(job?.state === "running" ? job.id : null);
}

/** 消えた実行を出していたら静電場に戻す */
export function useForgetRemovedRun(): void {
  const id = useResultsView((s) => s.activeRun);
  const exists = useJobs((s) => (id ? id in s.jobs : true));
  useEffect(() => {
    if (!exists) useResultsView.getState().setActiveRun(null);
  }, [exists]);
}

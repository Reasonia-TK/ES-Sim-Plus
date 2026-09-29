// /v2/jobs の REST。

import { ApiError, apiGet, apiPost } from "../backend/api";
import { baseUrl } from "../backend/port";
import type { Project } from "../model/project";
import type { JobKind, JobLimits, JobSummary } from "./types";

export const submitJob = (kind: JobKind, project: Project | null, options: Record<string, unknown> = {}, label?: string) =>
  apiPost<JobSummary>("/v2/jobs", { kind, project, options, label: label ?? null });

export const listJobs = () => apiGet<{ jobs: JobSummary[]; limits: JobLimits; kinds: string[] }>("/v2/jobs", { timeoutMs: 5000 });

export const stopJob = (id: string) => apiPost<JobSummary>(`/v2/jobs/${id}/stop`, {});

export const continueJob = (id: string, options: Record<string, unknown>) => apiPost<JobSummary>(`/v2/jobs/${id}/continue`, { options });

export async function deleteJob(id: string): Promise<void> {
  const path = `/v2/jobs/${id}`;
  const res = await fetch(baseUrl() + path, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw new ApiError(path, res.status, `HTTP ${res.status}`);
}

export const jobResult = <T = unknown>(id: string, signal?: AbortSignal) => apiGet<T>(`/v2/jobs/${id}/result`, { signal });

export const jobStarted = <T = unknown>(id: string) => apiGet<T>(`/v2/jobs/${id}/started`);

export const jobProject = (id: string) => apiGet<JobSummary & { project: Project | null }>(`/v2/jobs/${id}`);

export const sweepCase = <T = unknown>(id: string, i: number) => apiGet<T>(`/v2/jobs/${id}/cases/${i}`);

export async function setLimits(limits: { max_running?: number; per_kind?: Record<string, number> }): Promise<JobLimits> {
  const path = "/v2/jobs/limits";
  const res = await fetch(baseUrl() + path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(limits) });
  if (!res.ok) throw new ApiError(path, res.status, `HTTP ${res.status}`);
  return (await res.json()) as JobLimits;
}

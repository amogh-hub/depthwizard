export type StageState = "complete" | "active" | "pending" | "failed";

type Stage = {
  label: string;
  state: StageState;
  detail?: string;
};

export function StatusPipeline({ stages }: { stages: Stage[] }) {
  return (
    <div className="dw-pipeline">
      {stages.map((stage) => (
        <div className="dw-stage" data-state={stage.state} key={stage.label}>
          <span className="dw-stage-dot" />
          <span>{stage.label}</span>
          <small title={stage.detail}>{stage.detail ?? ""}</small>
        </div>
      ))}
    </div>
  );
}

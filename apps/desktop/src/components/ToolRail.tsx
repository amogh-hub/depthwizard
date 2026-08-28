import {
  CompareIcon,
  ExportIcon,
  MeasureIcon,
  ProfileIcon,
  ProjectIcon,
  StructureIcon,
  ValidateIcon,
} from "./icons";

const tools = [
  { id: "Project", label: "Navigate", icon: ProjectIcon },
  { id: "Measure", label: "Measure", icon: MeasureIcon },
  { id: "Structures", label: "Structures", icon: StructureIcon },
  { id: "Profiles", label: "Profiles", icon: ProfileIcon },
  { id: "Validation", label: "Validate", icon: ValidateIcon },
  { id: "Compare", label: "Compare", icon: CompareIcon },
  { id: "Export", label: "Export", icon: ExportIcon },
] as const;

export function ToolRail({
  active,
  onChange,
  disabledTools,
}: {
  active: string;
  onChange: (tool: string) => void;
  disabledTools?: ReadonlySet<string>;
}) {
  return (
    <nav className="dw-toolrail" aria-label="Workspace tools">
      {tools.map(({ id, label, icon: ToolIcon }) => {
        const disabled = disabledTools?.has(id) ?? false;
        return (
          <button
            key={id}
            className="dw-tool"
            data-active={active === id}
            title={disabled ? `${label} is unavailable until its required project evidence exists` : label}
            aria-label={label}
            disabled={disabled}
            onClick={() => {
              if (!disabled) onChange(id);
            }}
          >
            <span className="dw-tool-icon" aria-hidden="true"><ToolIcon /></span>
          </button>
        );
      })}
    </nav>
  );
}

import {
  CompareIcon,
  ExportIcon,
  LayersIcon,
  MeasureIcon,
  ProfileIcon,
  ProjectIcon,
  StructureIcon,
  ValidateIcon,
} from "./icons";

const tools = [
  ["Project", ProjectIcon],
  ["Layers", LayersIcon],
  ["Measure", MeasureIcon],
  ["Structures", StructureIcon],
  ["Profiles", ProfileIcon],
  ["Validation", ValidateIcon],
  ["Compare", CompareIcon],
  ["Export", ExportIcon],
] as const;

export function ToolRail({ active, onChange }: { active: string; onChange: (tool: string) => void }) {
  return (
    <nav className="dw-toolrail" aria-label="Workspace tools">
      {tools.map(([label, ToolIcon]) => (
        <button
          key={label}
          className="dw-tool"
          data-active={active === label}
          title={label}
          aria-label={label}
          onClick={() => onChange(label)}
        >
          <ToolIcon />
        </button>
      ))}
    </nav>
  );
}

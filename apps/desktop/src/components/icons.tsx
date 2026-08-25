import type { ReactNode } from "react";

const Icon = ({ children }: { children: ReactNode }) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

export const ProjectIcon = () => <Icon><path d="M4 7.5h6l1.7 2H20v9.5H4z"/><path d="M4 7.5V5h6l1.7 2"/></Icon>;
export const LayersIcon = () => <Icon><path d="m12 3 8 4-8 4-8-4 8-4Z"/><path d="m4 12 8 4 8-4"/><path d="m4 17 8 4 8-4"/></Icon>;
export const MeasureIcon = () => <Icon><path d="M4 17 17 4l3 3L7 20H4v-3Z"/><path d="m13.5 7.5 3 3"/></Icon>;
export const ProfileIcon = () => <Icon><path d="M3 18h18"/><path d="m5 16 4-6 3 3 4-8 3 11"/></Icon>;
export const ValidateIcon = () => <Icon><circle cx="12" cy="12" r="8"/><path d="m8.5 12 2.3 2.3 4.8-5"/></Icon>;
export const CompareIcon = () => <Icon><rect x="4" y="5" width="7" height="14" rx="1"/><rect x="13" y="5" width="7" height="14" rx="1"/></Icon>;
export const ExportIcon = () => <Icon><path d="M12 4v11"/><path d="m8 11 4 4 4-4"/><path d="M5 19h14"/></Icon>;
export const UploadIcon = () => <Icon><path d="M12 20V9"/><path d="m8 13 4-4 4 4"/><path d="M5 5h14"/></Icon>;

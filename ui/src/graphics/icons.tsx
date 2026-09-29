// ビューアの道具のアイコン (16 px、線は文字の色)。

import type { ReactNode } from "react";

function Svg({ children }: { children: ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

export const IconSelect = () => (
  <Svg>
    <path d="M3 2.5 L3 12.5 L6 9.8 L8.2 14 L9.9 13.2 L7.8 9.1 L11.8 9 Z" fill="currentColor" fillOpacity="0.15" />
  </Svg>
);

export const IconPolyline = () => (
  <Svg>
    <path d="M2.5 12.5 L5.5 4.5 L10 10 L13.5 3.5" />
    <circle cx="2.5" cy="12.5" r="1.2" fill="currentColor" />
    <circle cx="5.5" cy="4.5" r="1.2" fill="currentColor" />
    <circle cx="10" cy="10" r="1.2" fill="currentColor" />
    <circle cx="13.5" cy="3.5" r="1.2" fill="currentColor" />
  </Svg>
);

export const IconRect = () => (
  <Svg>
    <rect x="2.5" y="4" width="11" height="8" rx="0.5" />
  </Svg>
);

export const IconCircle = () => (
  <Svg>
    <circle cx="8" cy="8" r="5.5" />
  </Svg>
);

export const IconPlace = () => (
  <Svg>
    <path d="M2.5 13.5 L13.5 2.5" />
    <circle cx="2.5" cy="13.5" r="1.5" fill="currentColor" />
    <circle cx="13.5" cy="2.5" r="1.5" fill="currentColor" />
    <path d="M11 11 h3 M12.5 9.5 v3" />
  </Svg>
);

export const IconProbe = () => (
  <Svg>
    <circle cx="8" cy="8" r="3.5" />
    <path d="M8 1.5 v3 M8 11.5 v3 M1.5 8 h3 M11.5 8 h3" />
  </Svg>
);

export const IconMeasure = () => (
  <Svg>
    <path d="M1.8 10.5 L10.5 1.8 L14.2 5.5 L5.5 14.2 Z" />
    <path d="M4.5 7.8 l1.4 1.4 M6.6 5.7 l1 1 M8.7 3.6 l1.4 1.4" />
  </Svg>
);

export const IconSnap = () => (
  <Svg>
    <path d="M2 5 h12 M2 11 h12 M5 2 v12 M11 2 v12" strokeOpacity="0.55" />
    <circle cx="11" cy="11" r="1.8" fill="currentColor" />
  </Svg>
);

export const IconFit = () => (
  <Svg>
    <path d="M2 6 V2 H6 M10 2 H14 V6 M14 10 V14 H10 M6 14 H2 V10" />
    <rect x="5.5" y="5.5" width="5" height="5" strokeOpacity="0.6" />
  </Svg>
);

export const IconDisplay = () => (
  <Svg>
    <path d="M1.5 8 C3.5 4.2 12.5 4.2 14.5 8 C12.5 11.8 3.5 11.8 1.5 8 Z" />
    <circle cx="8" cy="8" r="2" />
  </Svg>
);

export const IconExport = () => (
  <Svg>
    <path d="M8 2 v8 M4.8 7 L8 10.2 L11.2 7" />
    <path d="M2.5 11.5 v2 h11 v-2" />
  </Svg>
);

export const IconChevron = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
    <path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);

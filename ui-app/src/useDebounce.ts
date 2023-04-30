import { useEffect } from "react";

const DEFAULT_DELAY = 1000;
export function useDebounce(effect: () => any, dependencies: any[], delay: number = DEFAULT_DELAY): void {
  useEffect(() => {
    const timeout = setTimeout(effect, delay);
    return () => clearTimeout(timeout);
  }, [delay, ...dependencies]);
}

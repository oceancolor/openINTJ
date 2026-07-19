import { InvalidArgumentError } from "commander";

/** Parse the explicit CLI A/B cohort without changing env-based defaults. */
export const parseProductBehaviorCohort = (value: string): boolean => {
  const cohort = value.trim().toLowerCase();
  if (cohort === "treatment" || cohort === "on" || cohort === "1") return true;
  if (cohort === "control" || cohort === "off" || cohort === "0") return false;
  throw new InvalidArgumentError("expected treatment|control (aliases: on|off, 1|0)");
};

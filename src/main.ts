import { mountCompetitionApp } from "./competition/app";

const root = document.querySelector<HTMLElement>("#app");
if (root) mountCompetitionApp(root);

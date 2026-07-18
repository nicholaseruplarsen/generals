import { mountApp } from "./ui/app";

const root = document.querySelector<HTMLElement>("#app");
if (root) mountApp(root);

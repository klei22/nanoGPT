import { createRoot } from "react-dom/client";
import { SphereForceLab } from "@/components/sphere-force-lab";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing app root.");
createRoot(root).render(<SphereForceLab />);

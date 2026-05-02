// Shiki transformer that recognizes step sentinel comments and tags lines
// with `data-step="N"`. Sentinel lines are stripped from the rendered output.
//
// Open:   // <step n=1>
// Close:  // </step>
//
// Lines between an open and the next close are tagged. Multiple open/close
// regions with the same `n` are allowed, so a step can map to non-contiguous
// ranges of lines.

const OPEN_RE = /^\/\/\s*<step\s+n=(\d+)>\s*$/;
const CLOSE_RE = /^\/\/\s*<\/step>\s*$/;

function getText(node) {
  if (!node) return "";
  if (node.type === "text") return node.value || "";
  if (node.children) return node.children.map(getText).join("");
  return "";
}

function isLineSpan(node) {
  if (!node || node.type !== "element" || node.tagName !== "span") return false;
  const cls = node.properties?.class ?? node.properties?.className;
  if (Array.isArray(cls)) return cls.includes("line");
  if (typeof cls === "string") return cls.split(/\s+/).includes("line");
  return false;
}

export function stepTransformer() {
  return {
    name: "step-tags",
    code(node) {
      const out = [];
      let currentStep = null;

      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];

        if (!isLineSpan(child)) {
          out.push(child);
          continue;
        }

        const text = getText(child).trim();

        const open = text.match(OPEN_RE);
        if (open) {
          currentStep = open[1];
          if (node.children[i + 1]?.type === "text") i++;
          continue;
        }
        if (CLOSE_RE.test(text)) {
          currentStep = null;
          if (node.children[i + 1]?.type === "text") i++;
          continue;
        }

        if (currentStep != null) {
          child.properties = child.properties || {};
          child.properties["data-step"] = currentStep;
        }
        out.push(child);
      }

      node.children = out;
    },
  };
}

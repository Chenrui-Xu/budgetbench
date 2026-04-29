const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Chenrui Xu, Hanyu Jin, Sylvia Li";
pres.title = "BudgetBench: Budget-Constrained Evaluation for Tool-Using Code Agents";

// ── Color Palette: Midnight Executive ────────────────────────────────
const C = {
  navy:      "1E2761",
  navyDark:  "151D4A",
  ice:       "CADCFC",
  white:     "FFFFFF",
  offWhite:  "F4F6FB",
  accent:    "4A90D9",
  accentLt:  "7BB3F0",
  green:     "27AE60",
  orange:    "E67E22",
  red:       "E74C3C",
  gray:      "7F8C9A",
  grayDark:  "4A5568",
  grayLight: "E2E8F0",
  purple:    "8E44AD",
  teal:      "1ABC9C",
};

// ── Reusable helpers ─────────────────────────────────────────────────
const RESULTS_KNAPSACK = path.resolve(__dirname, "../results/final");
const RESULTS_SWE = path.resolve(__dirname, "../results/swe_5issues_flash");

const TOTAL_SLIDES = 10;

function addFooter(slide, slideNum) {
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 5.15, w: 9.0, h: 0,
    line: { color: C.grayLight, width: 0.5 },
  });
  slide.addText("BudgetBench  |  CMU 24-880  |  Xu, Jin, Li", {
    x: 0.5, y: 5.2, w: 7, h: 0.35,
    fontSize: 9, color: C.gray, fontFace: "Calibri",
  });
  slide.addText(`${slideNum} / ${TOTAL_SLIDES}`, {
    x: 8.0, y: 5.2, w: 1.5, h: 0.35,
    fontSize: 9, color: C.gray, fontFace: "Calibri", align: "right",
  });
}

// =====================================================================
// SLIDE 1 — Title
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent },
  });

  slide.addText("BudgetBench", {
    x: 0.8, y: 1.2, w: 8.4, h: 1.0,
    fontSize: 44, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  slide.addText("Budget-Constrained Evaluation Harness\nfor Tool-Using Code Agents", {
    x: 0.8, y: 2.2, w: 8.4, h: 0.9,
    fontSize: 20, fontFace: "Calibri", color: C.ice, margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 3.3, w: 2.0, h: 0.04, fill: { color: C.accent },
  });

  slide.addText("Chenrui Xu  \u00b7  Hanyu Jin  \u00b7  Sylvia Li", {
    x: 0.8, y: 3.55, w: 8.4, h: 0.4,
    fontSize: 16, fontFace: "Calibri", color: C.ice, margin: 0,
  });

  slide.addText("CMU 24-880  AI Agents for Engineers  \u00b7  Spring 2026", {
    x: 0.8, y: 3.95, w: 8.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  slide.addText("April 21, 2026", {
    x: 0.8, y: 4.35, w: 8.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.gray, margin: 0,
  });
}

// =====================================================================
// SLIDE 2 — Problem & Motivation
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("The Problem", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 32, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  // ── Concrete example: same bug, two agents, different cost ──
  // Section label
  slide.addText("Same bug, two agents, same outcome:", {
    x: 0.6, y: 1.05, w: 8.8, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.grayDark, margin: 0,
  });

  // Agent A card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.5, w: 4.2, h: 1.6,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.5, w: 0.06, h: 1.6,
    fill: { color: C.green },
  });
  slide.addText("Agent A", {
    x: 0.85, y: 1.55, w: 2, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  slide.addText("5", {
    x: 0.85, y: 1.85, w: 1.2, h: 0.7,
    fontSize: 48, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });
  slide.addText("tool calls", {
    x: 2.0, y: 2.05, w: 1.5, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.grayDark, valign: "middle", margin: 0,
  });
  slide.addText("\u2714  Solved", {
    x: 3.5, y: 1.85, w: 1.2, h: 0.7,
    fontSize: 16, fontFace: "Calibri", color: C.green, bold: true, valign: "middle", margin: 0,
  });
  slide.addText("$0.01", {
    x: 0.85, y: 2.65, w: 3.8, h: 0.35,
    fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  // Agent B card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.5, w: 4.2, h: 1.6,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.5, w: 0.06, h: 1.6,
    fill: { color: C.red },
  });
  slide.addText("Agent B", {
    x: 5.45, y: 1.55, w: 2, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.red, bold: true, margin: 0,
  });
  slide.addText("50", {
    x: 5.45, y: 1.85, w: 1.5, h: 0.7,
    fontSize: 48, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });
  slide.addText("tool calls", {
    x: 6.8, y: 2.05, w: 1.5, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.grayDark, valign: "middle", margin: 0,
  });
  slide.addText("\u2714  Solved", {
    x: 8.1, y: 1.85, w: 1.2, h: 0.7,
    fontSize: 16, fontFace: "Calibri", color: C.green, bold: true, valign: "middle", margin: 0,
  });
  slide.addText("$0.50", {
    x: 5.45, y: 2.65, w: 3.8, h: 0.35,
    fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  // Equals sign between them
  slide.addText("vs.", {
    x: 4.4, y: 1.9, w: 1.2, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: C.gray, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // SWE-bench verdict
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 3.35, w: 8.8, h: 0.6,
    fill: { color: C.navy },
  });
  slide.addText([
    { text: "SWE-bench says:  ", options: { color: C.ice } },
    { text: "Both scored equally.  ", options: { bold: true, color: C.white } },
    { text: "Agent A's 10x efficiency is invisible.", options: { color: C.orange } },
  ], {
    x: 0.9, y: 3.35, w: 8.2, h: 0.6,
    fontSize: 15, fontFace: "Calibri", valign: "middle", margin: 0,
  });

  // Our solution
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.15, w: 8.8, h: 0.9,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.15, w: 0.06, h: 0.9,
    fill: { color: C.accent },
  });
  slide.addText("BudgetBench", {
    x: 0.85, y: 4.2, w: 2, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  slide.addText([
    { text: "Give each agent a fixed budget of B tool calls. Measure ", options: { breakLine: false } },
    { text: "Success@B", options: { bold: true, breakLine: false } },
    { text: ": can it solve the issue within budget? Agent A succeeds at B=5. Agent B needs B=50. Now we can tell the difference.", options: { breakLine: false } },
  ], {
    x: 0.85, y: 4.5, w: 8.2, h: 0.5,
    fontSize: 12.5, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.25,
  });

  addFooter(slide, 2);
}

// =====================================================================
// SLIDE 3 — Framework Architecture
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("BudgetBench Framework", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 32, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  // Flow diagram
  const boxH = 0.65, boxY = 1.4;

  // Agent
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: boxY, w: 1.8, h: boxH,
    fill: { color: C.accent },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 },
  });
  slide.addText("Agent", {
    x: 0.6, y: boxY, w: 1.8, h: boxH,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("\u2192", {
    x: 2.5, y: boxY, w: 0.5, h: boxH,
    fontSize: 22, color: C.gray, align: "center", valign: "middle", margin: 0,
  });

  // BudgetRunner
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.1, y: boxY, w: 2.2, h: boxH,
    fill: { color: C.navyDark },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 },
  });
  slide.addText("BudgetRunner", {
    x: 3.1, y: boxY, w: 2.2, h: boxH,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("\u2192", {
    x: 5.4, y: boxY, w: 0.5, h: boxH,
    fontSize: 22, color: C.gray, align: "center", valign: "middle", margin: 0,
  });

  // Environment
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.0, y: boxY, w: 2.0, h: boxH,
    fill: { color: C.green },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 },
  });
  slide.addText("Environment", {
    x: 6.0, y: boxY, w: 2.0, h: boxH,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("\u2192", {
    x: 8.1, y: boxY, w: 0.5, h: boxH,
    fontSize: 22, color: C.gray, align: "center", valign: "middle", margin: 0,
  });

  // Metrics
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 8.6, y: boxY, w: 1.0, h: boxH,
    fill: { color: C.purple },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.1 },
  });
  slide.addText("Metrics", {
    x: 8.6, y: boxY, w: 1.0, h: boxH,
    fontSize: 13, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Detail cards
  const cardY = 2.5, cardH = 2.4;

  // Card: BudgetRunner
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: cardY, w: 2.8, h: cardH,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: cardY, w: 0.06, h: cardH,
    fill: { color: C.navyDark },
  });
  slide.addText("BudgetRunner", {
    x: 0.85, y: cardY + 0.1, w: 2.4, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.navyDark, bold: true, margin: 0,
  });
  slide.addText([
    { text: "Intercepts every tool call", options: { breakLine: true } },
    { text: "Tracks cumulative cost", options: { breakLine: true } },
    { text: "Enforces budget ceiling", options: { breakLine: true } },
    { text: "Records full Episode trace", options: { breakLine: true } },
  ], {
    x: 0.85, y: cardY + 0.45, w: 2.4, h: 1.8,
    fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.5,
  });

  // Card: Environments
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.7, y: cardY, w: 2.8, h: cardH,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.7, y: cardY, w: 0.06, h: cardH,
    fill: { color: C.green },
  });
  slide.addText("Environments", {
    x: 3.95, y: cardY + 0.1, w: 2.4, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  slide.addText([
    { text: "Knapsack (toy)", options: { bold: true, breakLine: true } },
    { text: "  Heuristic agents, no LLM", options: { breakLine: true } },
    { text: "  Instant, deterministic", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "SWE-bench Lite (real)", options: { bold: true, breakLine: true } },
    { text: "  Docker sandbox per issue", options: { breakLine: true } },
    { text: "  Gemini API, eval images", options: { breakLine: true } },
  ], {
    x: 3.95, y: cardY + 0.45, w: 2.4, h: 1.8,
    fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.3,
  });

  // Card: Metrics
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: cardY, w: 2.8, h: cardH,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: cardY, w: 0.06, h: cardH,
    fill: { color: C.purple },
  });
  slide.addText("Metrics", {
    x: 7.05, y: cardY + 0.1, w: 2.4, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.purple, bold: true, margin: 0,
  });
  slide.addText([
    { text: "Success@B", options: { bold: true, breakLine: true } },
    { text: "  Solved within budget B?", options: { breakLine: true } },
    { text: "AUC (Success vs Budget)", options: { bold: true, breakLine: true } },
    { text: "  Primary ranking metric", options: { breakLine: true } },
    { text: "Violation rate", options: { bold: true, breakLine: true } },
    { text: "Bootstrap 95% CI", options: { bold: true, breakLine: true } },
  ], {
    x: 7.05, y: cardY + 0.45, w: 2.4, h: 1.8,
    fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.3,
  });

  addFooter(slide, 3);
}

// =====================================================================
// SLIDE 4 — LoopAgent: The Base Agent
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("LoopAgent: The Agentic Loop", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 32, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  slide.addText("Standard tool-using agent with no budget awareness", {
    x: 0.6, y: 0.9, w: 8.8, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  // Left: Loop flow diagram (vertical steps)
  const steps = [
    { label: "1. search", desc: "Find relevant files by keyword", color: C.accent },
    { label: "2. inspect", desc: "Read file contents (line ranges)", color: C.accent },
    { label: "3. edit", desc: "Apply a unified diff patch", color: C.accent },
    { label: "4. run_tests", desc: "Execute test suite (cost = 3x)", color: C.orange },
    { label: "5. submit_patch", desc: "Submit final patch for grading", color: C.green },
  ];

  const stepX = 0.6, stepW = 4.2;
  steps.forEach((s, i) => {
    const sy = 1.4 + i * 0.72;
    // Step box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: stepX, y: sy, w: stepW, h: 0.6,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.05 },
    });
    // Left color indicator
    slide.addShape(pres.shapes.RECTANGLE, {
      x: stepX, y: sy, w: 0.06, h: 0.6,
      fill: { color: s.color },
    });
    slide.addText(s.label, {
      x: stepX + 0.2, y: sy, w: 1.6, h: 0.6,
      fontSize: 13, fontFace: "Calibri", color: C.navyDark, bold: true,
      valign: "middle", margin: 0,
    });
    slide.addText(s.desc, {
      x: stepX + 1.8, y: sy, w: 2.2, h: 0.6,
      fontSize: 11, fontFace: "Calibri", color: C.grayDark,
      valign: "middle", margin: 0,
    });
    // Arrow down (except last)
    if (i < steps.length - 1) {
      slide.addText("\u2193", {
        x: stepX + 0.8, y: sy + 0.52, w: 0.5, h: 0.28,
        fontSize: 14, color: C.grayLight, align: "center", margin: 0,
      });
    }
  });

  // Loop-back arrow label
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.9, y: 1.4, w: 0.9, h: 2.88,
    fill: { color: "FFFFFF", transparency: 100 },
    line: { color: C.accent, width: 1.5, dashType: "dash" },
  });
  slide.addText("loop", {
    x: 3.95, y: 2.5, w: 0.8, h: 0.4,
    fontSize: 10, fontFace: "Calibri", color: C.accent, bold: true,
    align: "center", valign: "middle", margin: 0, rotate: 90,
  });

  // Right: Key characteristics
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.4, w: 4.2, h: 3.5,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.4, w: 4.2, h: 0.45,
    fill: { color: C.green },
  });
  slide.addText("How It Works", {
    x: 5.2, y: 1.4, w: 4.2, h: 0.45,
    fontSize: 14, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText([
    { text: "LLM backbone: ", options: { bold: true, breakLine: false } },
    { text: "Gemini 2.5 (Flash / Pro)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Each iteration: ", options: { bold: true, breakLine: false } },
    { text: "LLM sees issue + history, picks one tool call", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Tool costs: ", options: { bold: true, breakLine: false } },
    { text: "search/inspect/edit = 1, run_tests = 3", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Stopping: ", options: { bold: true, breakLine: false } },
    { text: "agent calls submit_patch, or budget runs out", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Problem: ", options: { bold: true, color: C.red, breakLine: false } },
    { text: "LoopAgent has no concept of remaining budget. It explores freely and often exhausts the limit before submitting.", options: { color: C.red, breakLine: true } },
  ], {
    x: 5.5, y: 2.05, w: 3.6, h: 2.7,
    fontSize: 12, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.2,
  });

  // SingleShot mention
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 4.95, w: 4.2, h: 0.0,
  });

  addFooter(slide, 4);
}

// =====================================================================
// SLIDE 5 — BudgetAwareAgent: Key Novelty
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("BudgetAwareAgent: Regime Switching", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 30, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  slide.addText("Extends LoopAgent with budget-adaptive strategy  |  Key Novelty", {
    x: 0.6, y: 0.9, w: 8.8, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  // Three regime cards
  const regimes = [
    {
      name: "HIGH",
      range: "> 60% remaining",
      color: C.green,
      strategy: "Thorough exploration",
      actions: "All tools allowed, including run_tests. Explore codebase, understand root cause.",
    },
    {
      name: "MEDIUM",
      range: "30\u201360% remaining",
      color: C.orange,
      strategy: "Focused execution",
      actions: "run_tests BLOCKED (costs 3x). Targeted search + edit only. Move toward submitting.",
    },
    {
      name: "LOW",
      range: "< 30% remaining",
      color: C.red,
      strategy: "Emergency submit",
      actions: "Skip LLM call entirely. Extract best patch from history and submit immediately.",
    },
  ];

  regimes.forEach((r, i) => {
    const rx = 0.6 + i * 3.1;
    const rw = 2.8, ry = 1.4, rh = 2.2;

    slide.addShape(pres.shapes.RECTANGLE, {
      x: rx, y: ry, w: rw, h: rh,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: rx, y: ry, w: rw, h: 0.06,
      fill: { color: r.color },
    });

    slide.addText(r.name, {
      x: rx + 0.15, y: ry + 0.15, w: 1.0, h: 0.35,
      fontSize: 18, fontFace: "Calibri", color: r.color, bold: true, margin: 0,
    });
    slide.addText(r.range, {
      x: rx + 1.1, y: ry + 0.15, w: 1.5, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.gray, valign: "middle", margin: 0,
    });
    slide.addText(r.strategy, {
      x: rx + 0.15, y: ry + 0.6, w: rw - 0.3, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.navyDark, bold: true, margin: 0,
    });
    slide.addText(r.actions, {
      x: rx + 0.15, y: ry + 0.95, w: rw - 0.3, h: 1.1,
      fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
      lineSpacingMultiple: 1.3,
    });
  });

  // Implementation box
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 3.85, w: 8.8, h: 1.15,
    fill: { color: C.navy },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 },
  });
  slide.addText("Implementation", {
    x: 0.9, y: 3.92, w: 3, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.accentLt, bold: true, margin: 0,
  });
  slide.addText([
    { text: "Inherits _BaseLoopAgent, overrides 3 hooks only (~120 lines of new code):", options: { breakLine: true } },
    { text: "_regime_section()", options: { bold: true, breakLine: false } },
    { text: " injects budget context into LLM prompt   |   ", options: { breakLine: false } },
    { text: "_pre_act()", options: { bold: true, breakLine: false } },
    { text: " short-circuits to submit in LOW   |   ", options: { breakLine: false } },
    { text: "_allow_run_tests()", options: { bold: true, breakLine: false } },
    { text: " blocks testing in MEDIUM", options: { breakLine: true } },
  ], {
    x: 0.9, y: 4.22, w: 8.2, h: 0.7,
    fontSize: 11, fontFace: "Calibri", color: C.ice, margin: 0,
    lineSpacingMultiple: 1.35,
  });

  // Also mention SingleShot
  slide.addText("Baseline: SingleShot agent uses 1 LLM call to produce a patch directly (no loop, no tools)", {
    x: 0.6, y: 5.05, w: 8.8, h: 0.0,
    fontSize: 0, fontFace: "Calibri", color: C.offWhite, margin: 0,
  });

  addFooter(slide, 5);
}

// =====================================================================
// SLIDE 6 — All Three Agents Overview
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("Agent Comparison", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 32, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  const agents = [
    {
      name: "SingleShot",
      color: C.accent,
      desc: "One LLM call, produce a patch immediately. No exploration, no iteration.",
      budget: "Cost: 1 tool call",
      pros: "Fast, cheap, 0% budget risk",
      cons: "Only works for simple bugs",
    },
    {
      name: "LoopAgent",
      color: C.green,
      desc: "Agentic loop: search, inspect, edit, test, repeat. Standard approach.",
      budget: "Cost: 6\u201318 tool calls",
      pros: "Can explore and verify fixes",
      cons: "No budget awareness; often exceeds limit",
    },
    {
      name: "BudgetAware",
      color: C.purple,
      desc: "Same loop as LoopAgent, but adapts strategy based on remaining budget.",
      budget: "Cost: 4\u201310 tool calls",
      pros: "Adapts to constraints; 0% violation rate",
      cons: "May submit prematurely in LOW regime",
    },
  ];

  agents.forEach((a, i) => {
    const cardX = 0.6 + i * 3.1;
    const cardW = 2.8, cardY2 = 1.15, cardH2 = 3.7;

    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: cardY2, w: cardW, h: cardH2,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 },
    });

    // Header
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: cardY2, w: cardW, h: 0.55,
      fill: { color: a.color },
    });
    slide.addText(a.name, {
      x: cardX, y: cardY2, w: cardW, h: 0.55,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });

    // Desc
    slide.addText(a.desc, {
      x: cardX + 0.2, y: cardY2 + 0.7, w: cardW - 0.4, h: 0.9,
      fontSize: 11.5, fontFace: "Calibri", color: C.grayDark, margin: 0,
      lineSpacingMultiple: 1.3,
    });

    // Budget badge
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cardX + 0.2, y: cardY2 + 1.65, w: cardW - 0.4, h: 0.35,
      fill: { color: C.offWhite },
    });
    slide.addText(a.budget, {
      x: cardX + 0.2, y: cardY2 + 1.65, w: cardW - 0.4, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: a.color, bold: true,
      align: "center", valign: "middle", margin: 0,
    });

    // Strength
    slide.addText("Strength", {
      x: cardX + 0.2, y: cardY2 + 2.2, w: cardW - 0.4, h: 0.25,
      fontSize: 10, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
    });
    slide.addText(a.pros, {
      x: cardX + 0.2, y: cardY2 + 2.45, w: cardW - 0.4, h: 0.45,
      fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
    });

    // Weakness
    slide.addText("Weakness", {
      x: cardX + 0.2, y: cardY2 + 2.95, w: cardW - 0.4, h: 0.25,
      fontSize: 10, fontFace: "Calibri", color: C.orange, bold: true, margin: 0,
    });
    slide.addText(a.cons, {
      x: cardX + 0.2, y: cardY2 + 3.2, w: cardW - 0.4, h: 0.45,
      fontSize: 11, fontFace: "Calibri", color: C.grayDark, margin: 0,
    });
  });

  addFooter(slide, 6);
}

// =====================================================================
// SLIDE 7 — Knapsack Results
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("Knapsack Results (450 Episodes)", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 30, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  slide.addText("50 seeds  \u00d7  3 budgets (5, 10, 20)  \u00d7  3 agents   |   Heuristic agents, no LLM", {
    x: 0.6, y: 0.9, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  slide.addImage({
    path: path.join(RESULTS_KNAPSACK, "success_curve.png"),
    x: 0.3, y: 1.35, w: 4.5, h: 2.8,
  });

  slide.addImage({
    path: path.join(RESULTS_KNAPSACK, "status_bars.png"),
    x: 5.0, y: 1.35, w: 4.5, h: 2.8,
  });

  // Key findings
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.25, w: 8.8, h: 0.8,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.06 },
  });
  slide.addText([
    { text: "Key finding: ", options: { bold: true, color: C.navyDark } },
    { text: "SingleShot and BudgetAware both reach 62% at all budgets. LoopAgent wastes budget on exploration \u2014 only 8% at B=20, all BUDGET_EXCEEDED at B=5,10. ", options: { color: C.grayDark } },
    { text: "BudgetAware matches SingleShot with 0% violations.", options: { bold: true, color: C.green } },
  ], {
    x: 0.85, y: 4.3, w: 8.3, h: 0.7,
    fontSize: 12, fontFace: "Calibri", margin: 0, lineSpacingMultiple: 1.2,
  });

  addFooter(slide, 7);
}

// =====================================================================
// SLIDE 8 — SWE-bench Results (Flash)
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("SWE-bench Results (Gemini Flash)", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 30, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  slide.addText("5 astropy issues  \u00d7  3 budgets (10, 20, 30)  \u00d7  3 agents   |   Docker sandbox + eval images", {
    x: 0.6, y: 0.9, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
  });

  slide.addImage({
    path: path.join(RESULTS_SWE, "success_curve.png"),
    x: 0.3, y: 1.35, w: 4.5, h: 2.8,
  });

  slide.addImage({
    path: path.join(RESULTS_SWE, "status_bars.png"),
    x: 5.0, y: 1.35, w: 4.5, h: 2.8,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.25, w: 8.8, h: 0.8,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.06 },
  });
  slide.addText([
    { text: "Key finding: ", options: { bold: true, color: C.navyDark } },
    { text: "LoopAgent reaches 40% at B=30 but 0% at B=10 (4/5 BUDGET_EXCEEDED). ", options: { color: C.grayDark } },
    { text: "BudgetAware achieves 20% at B=10 ", options: { bold: true, color: C.purple } },
    { text: "\u2014 the only agent to solve any issue at the lowest budget, with zero violations.", options: { color: C.grayDark } },
  ], {
    x: 0.85, y: 4.3, w: 8.3, h: 0.7,
    fontSize: 12, fontFace: "Calibri", margin: 0, lineSpacingMultiple: 1.2,
  });

  addFooter(slide, 8);
}

// =====================================================================
// SLIDE 9 — Limitations & Future Work
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.offWhite };

  slide.addText("Limitations & Future Work", {
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 30, fontFace: "Georgia", color: C.navyDark, bold: true, margin: 0,
  });

  // Limitations
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.2, w: 4.2, h: 3.6,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.2, w: 0.06, h: 3.6,
    fill: { color: C.orange },
  });
  slide.addText("Current Limitations", {
    x: 0.9, y: 1.3, w: 3.6, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.orange, bold: true, margin: 0,
  });
  slide.addText([
    { text: "1. Small SWE-bench sample (5 astropy issues)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "2. Budget = tool calls, not tokens or wall time", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "3. Fixed regime thresholds (0.6 / 0.3) not tuned", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "4. Single LLM backend (Gemini 2.5) tested", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "5. No multi-agent architectures compared", options: { breakLine: true } },
  ], {
    x: 0.9, y: 1.75, w: 3.6, h: 2.9,
    fontSize: 12, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.2,
  });

  // Future
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.2, w: 4.2, h: 3.6,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.06 },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.2, w: 0.06, h: 3.6,
    fill: { color: C.green },
  });
  slide.addText("Future Directions", {
    x: 5.5, y: 1.3, w: 3.6, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  slide.addText([
    { text: "1. Scale to full SWE-bench Lite (300 issues)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "2. Token-based budget (realistic cost model)", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "3. Learned thresholds via Bayesian optimization", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "4. Compare Claude, GPT-4, SWE-Agent, etc.", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "5. Dynamic budget allocation across subtasks", options: { breakLine: true } },
  ], {
    x: 5.5, y: 1.75, w: 3.6, h: 2.9,
    fontSize: 12, fontFace: "Calibri", color: C.grayDark, margin: 0,
    lineSpacingMultiple: 1.2,
  });

  addFooter(slide, 9);
}

// =====================================================================
// SLIDE 10 — Thank You / Q&A
// =====================================================================
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent },
  });

  slide.addText("Thank You", {
    x: 0.8, y: 1.0, w: 8.4, h: 1.0,
    fontSize: 44, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });

  slide.addText("Questions?", {
    x: 0.8, y: 1.9, w: 8.4, h: 0.6,
    fontSize: 24, fontFace: "Calibri", color: C.ice, margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 2.8, w: 2.0, h: 0.04, fill: { color: C.accent },
  });

  slide.addText([
    { text: "Project Highlights", options: { bold: true, fontSize: 14, breakLine: true, color: C.accentLt } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Framework: ", options: { bold: true, breakLine: false } },
    { text: "BudgetRunner + 2 environments + 4 metrics", options: { breakLine: true } },
    { text: "Agents: ", options: { bold: true, breakLine: false } },
    { text: "SingleShot, LoopAgent, BudgetAwareAgent", options: { breakLine: true } },
    { text: "Scale: ", options: { bold: true, breakLine: false } },
    { text: "450 Knapsack + 45 SWE-bench episodes", options: { breakLine: true } },
    { text: "Novelty: ", options: { bold: true, breakLine: false } },
    { text: "Budget-regime switching with 0% violation rate", options: { breakLine: true } },
  ], {
    x: 0.8, y: 3.0, w: 8.4, h: 2.0,
    fontSize: 13, fontFace: "Calibri", color: C.ice, margin: 0,
    lineSpacingMultiple: 1.5,
  });

  slide.addText("github.com/Eric-XC/budgetbench", {
    x: 0.8, y: 5.0, w: 8.4, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.gray, margin: 0,
  });
}

// ── Write file ───────────────────────────────────────────────────────
const outPath = path.resolve(__dirname, "BudgetBench_Presentation.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved:", outPath);
});

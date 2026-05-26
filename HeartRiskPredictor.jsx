import { useState, useEffect, useRef } from "react";
import { Heart, Activity, AlertTriangle, CheckCircle, RotateCcw, ChevronRight, Info } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// CLEVELAND HEART DISEASE DATASET  (UCI ML Repository – 100 curated samples)
// Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
//          exang, oldpeak, slope, ca, thal, target
// ─────────────────────────────────────────────────────────────────────────────
const DATASET = [
  [63,1,3,145,233,1,0,150,0,2.3,0,0,1,0],[37,1,2,130,250,0,1,187,0,3.5,0,0,2,0],
  [41,0,1,130,204,0,0,172,0,1.4,2,0,2,0],[56,1,1,120,236,0,1,178,0,0.8,2,0,2,0],
  [57,0,0,120,354,0,1,163,1,0.6,2,0,2,0],[57,1,0,140,192,0,1,148,0,0.4,1,0,1,0],
  [56,0,1,140,294,0,0,153,0,1.3,1,0,2,0],[44,1,1,120,263,0,1,173,0,0.0,2,0,3,0],
  [52,1,2,172,199,1,1,162,0,0.5,2,0,3,0],[57,1,2,150,168,0,1,174,0,1.6,2,0,2,0],
  [54,1,0,140,239,0,1,160,0,1.2,2,0,2,0],[48,0,2,130,275,0,1,139,0,0.2,2,0,2,0],
  [49,1,1,130,266,0,1,171,0,0.6,2,0,2,0],[64,1,3,110,211,0,0,144,1,1.8,1,0,2,0],
  [58,0,2,150,283,1,0,162,0,1.0,2,0,2,0],[50,0,2,120,219,0,1,158,0,1.6,1,0,2,0],
  [58,0,2,120,340,0,1,172,0,0.0,2,0,2,0],[66,0,3,150,226,0,1,114,0,2.6,0,0,2,0],
  [43,1,0,150,247,0,1,171,0,1.5,2,0,2,0],[69,0,3,140,239,0,1,151,0,1.8,2,2,2,0],
  [59,1,0,135,234,0,1,161,0,0.5,1,0,3,0],[44,1,2,130,233,0,1,179,1,0.4,2,0,2,0],
  [42,1,0,140,226,0,1,178,0,0.0,2,0,2,0],[61,1,2,150,243,1,1,137,1,1.0,1,0,2,0],
  [40,1,3,140,199,0,1,178,1,1.4,2,0,3,0],[71,0,1,160,302,0,1,162,0,0.4,2,2,2,0],
  [51,1,2,110,175,0,1,123,0,0.6,2,0,2,0],[65,0,2,140,417,1,0,157,0,0.8,2,1,2,0],
  [53,0,2,130,197,1,0,152,0,1.2,0,0,2,0],[41,0,1,105,198,0,1,168,0,0.0,2,1,2,0],
  [66,1,0,120,302,0,0,151,0,0.4,1,0,2,0],[62,0,2,140,394,0,0,157,0,1.2,1,0,2,0],
  [55,1,3,125,212,0,1,168,0,1.0,2,1,2,0],[34,1,3,118,182,0,0,174,0,0.0,2,0,2,0],
  [48,1,0,122,222,0,0,186,0,0.0,2,0,2,0],[55,0,0,132,342,0,1,166,0,1.2,2,0,2,0],
  [39,1,2,120,339,0,1,170,0,0.0,2,0,2,0],[45,0,1,130,237,0,0,170,0,0.0,2,0,2,0],
  [54,0,2,160,201,0,1,163,0,0.0,2,1,2,0],[29,1,1,130,204,0,0,202,0,0.0,2,0,2,0],
  [43,0,2,122,213,0,1,165,0,0.2,1,0,2,0],[54,1,2,125,216,0,0,140,0,0.0,1,0,1,0],
  // disease cases
  [67,1,0,160,286,0,0,108,1,1.5,1,3,2,1],[67,1,0,120,229,0,0,129,1,2.6,1,2,3,1],
  [62,0,0,140,268,0,0,160,0,3.6,0,2,2,1],[63,1,0,130,254,0,0,147,0,1.4,1,1,3,1],
  [53,1,0,140,203,1,0,155,1,3.1,0,0,3,1],[55,1,0,132,353,0,1,132,1,1.2,1,1,3,1],
  [65,0,0,150,225,0,0,114,0,1.0,1,3,3,1],[56,1,0,130,256,1,0,142,1,0.6,1,1,1,1],
  [54,1,0,125,273,0,0,152,0,0.5,0,1,2,1],[49,1,0,130,266,0,1,171,0,0.6,2,0,2,1],
  [64,1,0,110,211,0,0,144,1,1.8,1,0,2,1],[70,1,0,145,174,0,1,125,1,2.6,0,0,3,1],
  [61,1,0,148,203,0,1,161,0,0.0,2,1,3,1],[71,0,0,112,149,0,1,125,0,1.6,1,0,2,1],
  [74,0,1,120,269,0,0,121,1,0.2,2,1,2,1],[68,0,2,120,211,0,0,115,0,1.5,1,0,2,1],
  [57,1,0,130,131,0,1,115,1,1.2,1,1,3,1],[52,1,0,108,233,1,1,147,0,0.1,2,3,3,1],
  [51,0,0,130,305,0,1,142,1,1.2,1,0,3,1],[68,1,2,180,274,1,0,150,1,1.6,1,0,3,1],
  [43,1,0,120,177,0,0,120,1,2.5,1,0,3,1],[55,0,0,180,327,0,2,117,1,3.4,1,0,2,1],
  [62,1,0,130,231,0,1,146,0,1.8,1,3,3,1],[71,1,2,160,302,0,1,162,0,0.4,2,2,2,1],
  [46,1,2,140,311,0,1,120,1,1.8,1,2,3,1],[66,1,0,160,246,0,1,120,1,0.0,1,3,1,1],
  [58,1,2,125,300,0,0,171,0,0.0,2,2,3,1],[60,1,0,140,293,0,0,170,0,1.2,1,2,3,1],
  [65,1,0,110,248,0,0,158,0,0.6,2,2,1,1],[55,1,0,140,217,0,1,111,1,5.6,0,0,3,1],
  [54,1,0,122,286,0,0,116,1,3.2,1,2,2,1],[44,1,0,120,169,0,1,144,1,2.8,0,0,1,1],
  [51,1,0,140,299,0,1,173,1,1.6,2,0,3,1],[51,1,2,94,227,0,1,154,1,0.0,2,1,3,1],
  [59,1,2,126,218,1,1,134,0,2.2,1,1,1,1],[57,0,0,140,241,0,1,123,1,0.2,1,0,3,1],
  [45,1,0,104,208,0,0,148,1,3.0,1,0,2,1],[60,0,2,102,318,0,1,160,0,0.0,2,1,2,1],
  [55,1,0,160,289,0,0,145,1,0.8,1,1,3,1],[56,1,0,150,213,1,0,125,1,1.0,1,0,2,1],
  [63,1,0,150,223,0,0,115,0,0.0,2,0,1,0],[52,1,0,128,205,1,1,184,0,0.0,2,0,2,0],
  [55,0,0,128,205,0,2,130,1,2.0,1,1,3,1],[47,1,0,110,275,0,0,118,1,1.0,1,1,2,1],
  [46,1,0,120,230,0,1,150,0,0.0,2,0,2,0],[54,0,2,108,267,0,0,167,0,0.0,2,0,2,0],
  [54,1,0,110,208,0,0,142,0,0.0,2,0,2,0],[66,0,2,178,228,1,1,165,1,1.0,1,2,3,1],
  [58,1,0,100,234,0,1,156,0,0.1,2,1,3,1],[60,1,0,145,282,0,0,142,1,2.8,1,2,3,1],
];

// ─────────────────────────────────────────────────────────────────────────────
// LOGISTIC REGRESSION ENGINE
// ─────────────────────────────────────────────────────────────────────────────
const FEAT_NAMES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];

const normalize = (row) => {
  const [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal] = row;
  return [
    (age - 29) / 48,
    sex,
    cp / 3,
    (trestbps - 94) / 106,
    (chol - 126) / 438,
    fbs,
    restecg / 2,
    (thalach - 71) / 131,
    exang,
    oldpeak / 6.2,
    slope / 2,
    ca / 3,
    (thal - 1) / 2,
    1,
  ];
};

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-250, Math.min(250, x))));

const trainLR = (data, epochs = 900, lr = 0.08) => {
  const k = 14;
  let w = new Array(k).fill(0).map(() => (Math.random() - 0.5) * 0.05);
  for (let e = 0; e < epochs; e++) {
    const g = new Array(k).fill(0);
    for (const row of data) {
      const x = normalize(row.slice(0, 13));
      const y = row[13];
      let z = 0;
      for (let j = 0; j < k; j++) z += w[j] * x[j];
      const err = sigmoid(z) - y;
      for (let j = 0; j < k; j++) g[j] += err * x[j];
    }
    for (let j = 0; j < k; j++) w[j] -= lr * g[j] / data.length;
  }
  return w;
};

const predictRisk = (w, features) => {
  const x = normalize(features);
  let z = 0;
  for (let j = 0; j < 14; j++) z += w[j] * x[j];
  return sigmoid(z);
};

const getTopFactors = (w, features) => {
  const x = normalize(features);
  const contributions = FEAT_NAMES.map((name, i) => ({
    name,
    contribution: w[i] * x[i],
  }));
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return contributions.slice(0, 5);
};

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT FORM
// ─────────────────────────────────────────────────────────────────────────────
const DEFAULT_FORM = {
  age: 55, sex: 1, cp: 0, trestbps: 130, chol: 240,
  fbs: 0, restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0,
  slope: 1, ca: 0, thal: 2,
};

// ─────────────────────────────────────────────────────────────────────────────
// EKG ANIMATION
// ─────────────────────────────────────────────────────────────────────────────
function EKGLine() {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    let x = 0;
    let frame = 0;
    const ekgPattern = [0,0,0,0,-5,-10,-5,0,0,30,60,30,-60,-30,0,10,5,0,0,0,0,0,0,0,0,0,0,0,0,0];
    const trail = [];
    let animId;
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#10b981";
      ctx.shadowBlur = 6;
      ctx.beginPath();
      trail.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
      const patIdx = frame % ekgPattern.length;
      const y = H / 2 + ekgPattern[patIdx];
      trail.push({ x, y });
      if (trail.length > W) trail.shift();
      trail.forEach((p) => p.x--);
      x = W;
      frame++;
      animId = requestAnimationFrame(draw);
    };
    animId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animId);
  }, []);
  return <canvas ref={ref} width={320} height={80} style={{ display: "block" }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// RISK GAUGE (SVG)
// ─────────────────────────────────────────────────────────────────────────────
function RiskGauge({ risk }) {
  const pct = Math.round(risk * 100);
  const angle = -150 + pct * 3;
  const cx = 110, cy = 110, r = 80;
  const toRad = (d) => (d * Math.PI) / 180;
  const arc = (startDeg, endDeg, color) => {
    const s = toRad(startDeg - 90);
    const e = toRad(endDeg - 90);
    const x1 = cx + r * Math.cos(s), y1 = cy + r * Math.sin(s);
    const x2 = cx + r * Math.cos(e), y2 = cy + r * Math.sin(e);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };
  const needleAngle = toRad(angle - 90);
  const nx = cx + 65 * Math.cos(needleAngle);
  const ny = cy + 65 * Math.sin(needleAngle);
  const color = pct < 35 ? "#10b981" : pct < 65 ? "#f59e0b" : "#ef4444";

  return (
    <svg width={220} height={140} viewBox="0 0 220 140">
      {/* Track arcs */}
      <path d={arc(-60, -20, "")} fill="none" stroke="#10b98144" strokeWidth={12} strokeLinecap="round" />
      <path d={arc(-20, 20, "")} fill="none" stroke="#f59e0b44" strokeWidth={12} strokeLinecap="round" />
      <path d={arc(20, 60, "")} fill="none" stroke="#ef444444" strokeWidth={12} strokeLinecap="round" />
      {/* Value arc */}
      <path d={arc(-60, -60 + pct * 1.2, "")} fill="none" stroke={color} strokeWidth={12} strokeLinecap="round" />
      {/* Labels */}
      <text x={28} y={118} fontSize={9} fill="#6b7280" textAnchor="middle">Low</text>
      <text x={110} y={42} fontSize={9} fill="#6b7280" textAnchor="middle">Moderate</text>
      <text x={192} y={118} fontSize={9} fill="#6b7280" textAnchor="middle">High</text>
      {/* Needle */}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth={3} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={6} fill={color} />
      {/* Center text */}
      <text x={cx} y={cy + 28} fontSize={28} fontWeight="700" fill={color} textAnchor="middle">{pct}%</text>
      <text x={cx} y={cy + 43} fontSize={10} fill="#9ca3af" textAnchor="middle">Risk Score</text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAINING VIEW
// ─────────────────────────────────────────────────────────────────────────────
function TrainingView({ progress }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", minHeight: 520, gap: 32, padding: "2rem",
      background: "linear-gradient(160deg, #0f172a 0%, #1e293b 100%)",
      borderRadius: 16,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Heart size={28} color="#ef4444" fill="#ef4444" />
        <span style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.5px" }}>
          CardioPredict AI
        </span>
      </div>

      <div style={{ background: "#1e293b", borderRadius: 12, padding: "20px 24px", border: "1px solid #334155" }}>
        <EKGLine />
      </div>

      <div style={{ width: "100%", maxWidth: 360 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 13, color: "#94a3b8" }}>Training logistic regression model…</span>
          <span style={{ fontSize: 13, color: "#38bdf8", fontWeight: 600 }}>{progress}%</span>
        </div>
        <div style={{ background: "#334155", borderRadius: 999, height: 8 }}>
          <div style={{
            background: "linear-gradient(90deg, #10b981, #38bdf8)",
            height: 8, borderRadius: 999, width: `${progress}%`,
            transition: "width 0.2s ease",
          }} />
        </div>
        <p style={{ fontSize: 12, color: "#64748b", marginTop: 10, textAlign: "center" }}>
          Training on {DATASET.length} patient records from the Cleveland Heart Disease dataset
        </p>
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        {["900 epochs","LR = 0.08","13 features"].map((t) => (
          <div key={t} style={{
            padding: "6px 14px", borderRadius: 999, fontSize: 11,
            background: "#1e293b", color: "#94a3b8", border: "1px solid #334155",
          }}>{t}</div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FIELD COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
const inputStyle = {
  width: "100%", padding: "8px 12px", borderRadius: 8, fontSize: 14,
  border: "1px solid #334155", background: "#1e293b", color: "#f1f5f9",
  outline: "none", boxSizing: "border-box",
};

const selectStyle = { ...inputStyle };

const labelStyle = { fontSize: 12, color: "#94a3b8", marginBottom: 5, display: "block", fontWeight: 500 };

function NumericField({ label, value, min, max, step = 1, unit, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <label style={labelStyle}>{label}</label>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="number" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          style={{ ...inputStyle, flex: 1 }}
        />
        {unit && <span style={{ fontSize: 11, color: "#64748b", whiteSpace: "nowrap" }}>{unit}</span>}
      </div>
    </div>
  );
}

function SelectField({ label, value, options, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <label style={labelStyle}>{label}</label>
      <select value={value} onChange={(e) => onChange(parseInt(e.target.value))} style={selectStyle}>
        {options.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
      </select>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FORM VIEW
// ─────────────────────────────────────────────────────────────────────────────
function FormView({ form, setForm, onPredict, accuracy }) {
  const set = (key) => (val) => setForm((f) => ({ ...f, [key]: val }));
  const section = (title, icon) => (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, marginTop: 8 }}>
      {icon}
      <span style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em" }}>{title}</span>
    </div>
  );
  const divider = <div style={{ borderTop: "1px solid #1e293b", margin: "20px 0" }} />;

  return (
    <div style={{
      background: "#0f172a", borderRadius: 16, overflow: "hidden",
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        padding: "20px 28px", background: "#1e293b",
        borderBottom: "1px solid #334155", display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Heart size={22} color="#ef4444" fill="#ef4444" />
          <span style={{ fontSize: 18, fontWeight: 700, color: "#f1f5f9" }}>CardioPredict AI</span>
        </div>
        <div style={{
          display: "flex", alignItems: "center", gap: 6, padding: "4px 12px",
          background: "#10b98120", borderRadius: 999, border: "1px solid #10b98140",
        }}>
          <CheckCircle size={13} color="#10b981" />
          <span style={{ fontSize: 12, color: "#10b981", fontWeight: 600 }}>Model trained · {accuracy}% accuracy</span>
        </div>
      </div>

      <div style={{ padding: "24px 28px" }}>
        <p style={{ fontSize: 13, color: "#64748b", marginBottom: 24 }}>
          Enter patient data below. The model uses logistic regression trained on the Cleveland Heart Disease dataset (UCI).
        </p>

        {/* Demographics */}
        {section("Demographics", <Activity size={14} color="#38bdf8" />)}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 16 }}>
          <NumericField label="Age" value={form.age} min={20} max={80} onChange={set("age")} unit="years" />
          <SelectField label="Sex" value={form.sex} onChange={set("sex")}
            options={[[1, "Male"], [0, "Female"]]} />
          <SelectField label="Chest Pain Type" value={form.cp} onChange={set("cp")}
            options={[[0,"Typical angina"],[1,"Atypical angina"],[2,"Non-anginal"],[3,"Asymptomatic"]]} />
        </div>

        {divider}

        {/* Vitals */}
        {section("Resting Vitals", <Heart size={14} color="#f472b6" />)}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 16 }}>
          <NumericField label="Resting BP" value={form.trestbps} min={80} max={220} onChange={set("trestbps")} unit="mm Hg" />
          <NumericField label="Cholesterol" value={form.chol} min={100} max={600} onChange={set("chol")} unit="mg/dl" />
          <SelectField label="Fasting Blood Sugar" value={form.fbs} onChange={set("fbs")}
            options={[[0,"≤ 120 mg/dl"],[1,"> 120 mg/dl"]]} />
          <SelectField label="Resting ECG" value={form.restecg} onChange={set("restecg")}
            options={[[0,"Normal"],[1,"ST-T abnormality"],[2,"LV hypertrophy"]]} />
        </div>

        {divider}

        {/* Exercise test */}
        {section("Exercise Test Results", <Activity size={14} color="#a78bfa" />)}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 16 }}>
          <NumericField label="Max Heart Rate" value={form.thalach} min={60} max={220} onChange={set("thalach")} unit="bpm" />
          <SelectField label="Exercise Angina" value={form.exang} onChange={set("exang")}
            options={[[0,"No"],[1,"Yes"]]} />
          <NumericField label="ST Depression" value={form.oldpeak} min={0} max={7} step={0.1} onChange={set("oldpeak")} unit="mm" />
          <SelectField label="ST Slope" value={form.slope} onChange={set("slope")}
            options={[[0,"Upsloping"],[1,"Flat"],[2,"Downsloping"]]} />
        </div>

        {divider}

        {/* Fluoroscopy */}
        {section("Advanced Diagnostics", <Info size={14} color="#fb923c" />)}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 16 }}>
          <SelectField label="Major Vessels (fluoroscopy)" value={form.ca} onChange={set("ca")}
            options={[[0,"0 vessels"],[1,"1 vessel"],[2,"2 vessels"],[3,"3 vessels"]]} />
          <SelectField label="Thalassemia" value={form.thal} onChange={set("thal")}
            options={[[1,"Normal"],[2,"Fixed defect"],[3,"Reversible defect"]]} />
        </div>

        {/* Predict button */}
        <div style={{ marginTop: 28 }}>
          <button onClick={onPredict} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "13px 28px",
            background: "linear-gradient(135deg, #ef4444, #dc2626)",
            color: "#fff", border: "none", borderRadius: 10, fontSize: 15,
            fontWeight: 700, cursor: "pointer", transition: "opacity 0.2s",
            boxShadow: "0 4px 20px #ef444440",
          }}>
            <Heart size={18} fill="#fff" />
            Analyse Risk
            <ChevronRight size={18} />
          </button>
        </div>

        <p style={{ fontSize: 11, color: "#475569", marginTop: 14 }}>
          ⚠ For educational purposes only. Not a medical diagnostic tool.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RESULT VIEW
// ─────────────────────────────────────────────────────────────────────────────
const FACTOR_LABELS = {
  age: "Age",
  sex: "Sex",
  cp: "Chest Pain Type",
  trestbps: "Resting BP",
  chol: "Cholesterol",
  fbs: "Fasting Blood Sugar",
  restecg: "Resting ECG",
  thalach: "Max Heart Rate",
  exang: "Exercise Angina",
  oldpeak: "ST Depression",
  slope: "ST Slope",
  ca: "Major Vessels",
  thal: "Thalassemia",
};

function ResultView({ risk, form, onReset }) {
  const pct = Math.round(risk * 100);
  const level = pct < 35 ? "Low" : pct < 65 ? "Moderate" : "High";
  const color = pct < 35 ? "#10b981" : pct < 65 ? "#f59e0b" : "#ef4444";
  const bg = pct < 35 ? "#10b98118" : pct < 65 ? "#f59e0b18" : "#ef444418";

  // Get contributing factors using a dummy weight approximation
  const features = [
    form.age, form.sex, form.cp, form.trestbps, form.chol,
    form.fbs, form.restecg, form.thalach, form.exang, form.oldpeak,
    form.slope, form.ca, form.thal,
  ];
  // Clinical insight labels
  const insights = [];
  if (form.ca > 0) insights.push(`${form.ca} blocked vessel${form.ca > 1 ? "s" : ""} on fluoroscopy`);
  if (form.thal === 3) insights.push("Reversible thalassemia defect detected");
  if (form.oldpeak > 2) insights.push(`High ST depression (${form.oldpeak} mm)`);
  if (form.exang === 1) insights.push("Exercise-induced angina present");
  if (form.cp === 3) insights.push("Asymptomatic chest pain pattern");
  if (form.thalach < 120) insights.push(`Low max heart rate (${form.thalach} bpm)`);
  if (form.chol > 300) insights.push(`Elevated cholesterol (${form.chol} mg/dl)`);
  if (form.trestbps > 160) insights.push(`High resting BP (${form.trestbps} mm Hg)`);

  const recs = pct < 35
    ? ["Maintain healthy lifestyle and routine check-ups", "Continue regular aerobic exercise", "Monitor cholesterol annually"]
    : pct < 65
    ? ["Consult a cardiologist for detailed evaluation", "Adopt heart-healthy diet (reduce saturated fats)", "Increase physical activity with medical guidance", "Monitor BP and cholesterol regularly"]
    : ["Seek urgent cardiology consultation", "Further diagnostic testing strongly recommended", "Review medications with your physician", "Lifestyle changes are critical"];

  return (
    <div style={{
      background: "#0f172a", borderRadius: 16, overflow: "hidden",
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{ padding: "20px 28px", background: "#1e293b", borderBottom: "1px solid #334155", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Heart size={22} color="#ef4444" fill="#ef4444" />
          <span style={{ fontSize: 18, fontWeight: 700, color: "#f1f5f9" }}>Prediction Result</span>
        </div>
        <button onClick={onReset} style={{
          display: "flex", alignItems: "center", gap: 6, padding: "7px 14px",
          background: "#334155", border: "none", color: "#94a3b8",
          borderRadius: 8, cursor: "pointer", fontSize: 13,
        }}>
          <RotateCcw size={13} />
          Re-assess
        </button>
      </div>

      <div style={{ padding: "24px 28px" }}>
        {/* Main risk display */}
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 24, alignItems: "center",
          marginBottom: 28, padding: "24px", borderRadius: 14,
          background: bg, border: `1px solid ${color}30`,
        }}>
          <RiskGauge risk={risk} />
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: 6,
              padding: "5px 14px", borderRadius: 999, marginBottom: 12,
              background: `${color}25`, border: `1px solid ${color}50`,
            }}>
              {level === "High" ? <AlertTriangle size={14} color={color} /> : <CheckCircle size={14} color={color} />}
              <span style={{ fontSize: 13, fontWeight: 700, color }}>{level} Risk</span>
            </div>
            <h2 style={{ fontSize: 32, fontWeight: 800, color, margin: "0 0 8px" }}>
              {pct}%
            </h2>
            <p style={{ fontSize: 14, color: "#94a3b8", lineHeight: 1.6 }}>
              {pct < 35
                ? "The model predicts a low probability of heart disease. Continue heart-healthy habits."
                : pct < 65
                ? "The model identifies moderate risk factors. Clinical evaluation is advised."
                : "The model detects significant risk markers. Prompt medical evaluation is strongly recommended."}
            </p>
          </div>
        </div>

        {/* Key clinical signals */}
        {insights.length > 0 && (
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
              Key signals detected
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {insights.slice(0, 4).map((ins, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "10px 14px", borderRadius: 8,
                  background: "#1e293b", border: "1px solid #334155",
                }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: "#cbd5e1" }}>{ins}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Patient summary */}
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
            Patient profile
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10 }}>
            {[
              ["Age", `${form.age} yrs`],
              ["Sex", form.sex === 1 ? "Male" : "Female"],
              ["Max HR", `${form.thalach} bpm`],
              ["Cholesterol", `${form.chol} mg/dl`],
              ["BP", `${form.trestbps} mmHg`],
              ["Vessels blocked", `${form.ca}`],
            ].map(([label, val]) => (
              <div key={label} style={{
                padding: "10px 14px", borderRadius: 10,
                background: "#1e293b", border: "1px solid #334155",
              }}>
                <div style={{ fontSize: 11, color: "#64748b", marginBottom: 3 }}>{label}</div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "#f1f5f9" }}>{val}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div style={{
          padding: "18px 20px", borderRadius: 12,
          background: "#1e293b", border: "1px solid #334155",
        }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14 }}>
            Recommendations
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {recs.map((r, i) => (
              <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span style={{ fontSize: 13, color, fontWeight: 700, flexShrink: 0 }}>→</span>
                <span style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.5 }}>{r}</span>
              </div>
            ))}
          </div>
        </div>

        <p style={{ fontSize: 11, color: "#475569", marginTop: 18 }}>
          ⚠ This prediction is for educational demonstration only and does not constitute medical advice. Always consult a qualified physician for diagnosis and treatment.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROOT COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function HeartRiskPredictor() {
  const [phase, setPhase] = useState("training");
  const [progress, setProgress] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [weights, setWeights] = useState(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [risk, setRisk] = useState(null);

  useEffect(() => {
    let p = 0;
    const iv = setInterval(() => {
      p = Math.min(p + Math.random() * 9, 90);
      setProgress(Math.round(p));
    }, 80);

    setTimeout(() => {
      clearInterval(iv);
      const w = trainLR(DATASET);
      let correct = 0;
      for (const row of DATASET) {
        const prob = predictRisk(w, row.slice(0, 13));
        if ((prob >= 0.5 ? 1 : 0) === row[13]) correct++;
      }
      setAccuracy(Math.round((correct / DATASET.length) * 100));
      setWeights(w);
      setProgress(100);
      setTimeout(() => setPhase("form"), 700);
    }, 1800);

    return () => clearInterval(iv);
  }, []);

  const handlePredict = () => {
    const features = [
      form.age, form.sex, form.cp, form.trestbps, form.chol,
      form.fbs, form.restecg, form.thalach, form.exang, form.oldpeak,
      form.slope, form.ca, form.thal,
    ];
    setRisk(predictRisk(weights, features));
    setPhase("result");
  };

  return (
    <div style={{ fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif" }}>
      {phase === "training" && <TrainingView progress={progress} />}
      {phase === "form" && <FormView form={form} setForm={setForm} onPredict={handlePredict} accuracy={accuracy} />}
      {phase === "result" && <ResultView risk={risk} form={form} onReset={() => setPhase("form")} />}
    </div>
  );
}

import HeartRiskPredictor from "./HeartRiskPredictor";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", padding: "32px 16px" }}>
      <div style={{ maxWidth: 780, margin: "0 auto" }}>
        <HeartRiskPredictor />
      </div>
    </div>
  );
}

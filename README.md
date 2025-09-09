# 🚀 Quant Strategy Generator & Allocator

This project is an **end-to-end automated quant trading pipeline**.  
It doesn’t just backtest one strategy — it **creates**, **allocates**, and **evaluates** entire portfolios of strategies using AI.

---

## ✨ What It Does

1. **Strategy Generation** 🧬  
   - A custom neural generator (FGEG) invents trading strategies.  
   - Instead of a human tweaking indicators, the system learns profitable patterns directly.  

2. **Capital Allocation** 📊  
   - A reinforcement learning agent (PPO) decides how much capital each strategy gets.  
   - Ensures diversification, risk control, and max returns.  

3. **Evaluation** ✅  
   - Final portfolio is stress-tested on unseen market data.  
   - Reports total PnL, annualized returns, and drawdowns.  

---

## 🖼️ How It Works

<p align="center">
  
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280"/>
    </marker>
  </defs>
  <style>
    .box { fill:#1f2937; stroke:#4f46e5; stroke-width:2; rx:12; ry:12; }
    .subbox { fill:#374151; stroke:#60a5fa; stroke-width:1.5; rx:8; ry:8; }
    .label { fill:white; font-family:Arial, sans-serif; font-size:15px; font-weight:bold; text-anchor:middle; }
    .sublabel { fill:#d1d5db; font-family:Arial, sans-serif; font-size:13px; text-anchor:middle; }
    .arrow { stroke:#6b7280; stroke-width:2.5; marker-end:url(#arrowhead); }
  </style>

  <!-- Input -->
  <rect x="40" y="200" width="180" height="80" class="box"/>
  <text x="130" y="230" class="label">📥 Random Noise</text>
  <text x="130" y="250" class="sublabel">(seed vector)</text>

  <!-- Strategy Generator -->
  <rect x="260" y="130" width="240" height="220" class="box"/>
  <text x="380" y="160" class="label">🧬 Strategy Generator</text>
  <text x="380" y="180" class="sublabel">(FGEG)</text>
  <rect x="280" y="200" width="200" height="40" class="subbox"/>
  <text x="380" y="225" class="sublabel">Generates Strategy Vectors</text>
  <rect x="280" y="250" width="200" height="40" class="subbox"/>
  <text x="380" y="275" class="sublabel">Denormalization</text>

  <!-- Backtester -->
  <rect x="540" y="80" width="240" height="120" class="box"/>
  <text x="660" y="110" class="label">📈 Backtester</text>
  <text x="660" y="135" class="sublabel">PnL • Sharpe • Drawdown</text>

  <!-- Fitness -->
  <rect x="540" y="250" width="240" height="120" class="box"/>
  <text x="660" y="280" class="label">⚖️ Fitness Function</text>
  <text x="660" y="305" class="sublabel">Scores strategies</text>

  <!-- Selection -->
  <rect x="820" y="160" width="180" height="80" class="box"/>
  <text x="910" y="195" class="label">🏆 Elite Strategies</text>

  <!-- PPO Allocator -->
  <rect x="1040" y="120" width="240" height="120" class="box"/>
  <text x="1160" y="150" class="label">🤖 PPO Allocator</text>
  <text x="1160" y="175" class="sublabel">Learns capital weights</text>

  <!-- Evaluation -->
  <rect x="1040" y="280" width="240" height="120" class="box"/>
  <text x="1160" y="310" class="label">✅ Final Evaluation</text>
  <text x="1160" y="335" class="sublabel">Out-of-sample backtest</text>
  <text x="1160" y="360" class="sublabel">Performance Report</text>

  <!-- Arrows -->
  <line x1="220" y1="240" x2="260" y2="240" class="arrow"/>
  <line x1="500" y1="190" x2="540" y2="140" class="arrow"/>
  <line x1="500" y1="240" x2="540" y2="290" class="arrow"/>
  <line x1="780" y1="190" x2="820" y2="190" class="arrow"/>
  <line x1="1000" y1="190" x2="1040" y2="190" class="arrow"/>
  <line x1="1000" y1="310" x2="1040" y2="310" class="arrow"/>
</svg>

</p>

---

## ⚡ Quickstart

```bash
git clone https://github.com/yourusername/quant-strategy-gen.git
cd quant-strategy-gen
python main.py

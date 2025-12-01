"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

type HistoryLap = {
  lap: string;
  lapTimeMs: string;
};

type FuturePit = {
  lap: string;
  durationMs: string;
};

type ScenarioForm = {
  name: string;
  futurePits: FuturePit[];
};

type ScenarioResult = {
  name: string;
  successProb: number;
  futurePits: { lap: number; durationMs: number }[];
  relativeScore?: number | null;
  rating?: number | null;
};

type StrategyResponse = {
  bestScenario: ScenarioResult;
  allScenarios: ScenarioResult[];
};

const GRAND_PRIX_OPTIONS = [
  { value: "Australian Grand Prix", label: "Australian Grand Prix" },
  { value: "Bahrain Grand Prix", label: "Bahrain Grand Prix" },
  { value: "Chinese Grand Prix", label: "Chinese Grand Prix" },
  { value: "Azerbaijan Grand Prix", label: "Azerbaijan Grand Prix" },
  { value: "Miami Grand Prix", label: "Miami Grand Prix" },
  { value: "Monaco Grand Prix", label: "Monaco Grand Prix" },
  { value: "Spanish Grand Prix", label: "Spanish Grand Prix" },
  { value: "Canadian Grand Prix", label: "Canadian Grand Prix" },
  { value: "Austrian Grand Prix", label: "Austrian Grand Prix" },
  { value: "British Grand Prix", label: "British Grand Prix" },
  { value: "Hungarian Grand Prix", label: "Hungarian Grand Prix" },
  { value: "Belgian Grand Prix", label: "Belgian Grand Prix" },
  { value: "Dutch Grand Prix", label: "Dutch Grand Prix" },
  { value: "Italian Grand Prix", label: "Italian Grand Prix" },
  { value: "Singapore Grand Prix", label: "Singapore Grand Prix" },
  { value: "Japanese Grand Prix", label: "Japanese Grand Prix" },
  { value: "Qatar Grand Prix", label: "Qatar Grand Prix" },
  { value: "United States Grand Prix", label: "United States Grand Prix" },
  { value: "Mexican Grand Prix", label: "Mexican Grand Prix" },
  { value: "São Paulo Grand Prix", label: "São Paulo Grand Prix" },
  { value: "Las Vegas Grand Prix", label: "Las Vegas Grand Prix" },
  { value: "Abu Dhabi Grand Prix", label: "Abu Dhabi Grand Prix" },
];

const DRIVER_OPTIONS = [
  { value: "HAM", label: "HAM – Lewis Hamilton" },
  { value: "HUL", label: "HUL – Nico Hülkenberg" },
  { value: "KUB", label: "KUB – Robert Kubica" },
  { value: "BOT", label: "BOT – Valtteri Bottas" },
  { value: "VER", label: "VER – Max Verstappen" },
  { value: "LEC", label: "LEC – Charles Leclerc" },
];

export default function StrategyPage() {
  const [grandPrix, setGrandPrix] = useState("Monaco Grand Prix");
  const [driverCode, setDriverCode] = useState("VER");
  const [cutLap, setCutLap] = useState("35");

  const [history, setHistory] = useState<HistoryLap[]>([
    { lap: "30", lapTimeMs: "75500" },
    { lap: "32", lapTimeMs: "76200" },
    { lap: "34", lapTimeMs: "76800" },
  ]);

  const [scenarios, setScenarios] = useState<ScenarioForm[]>([
    { name: "Aggressive Undercut", futurePits: [{ lap: "38", durationMs: "23500" }] },
    { name: "Conservative 1-Stop", futurePits: [{ lap: "40", durationMs: "24000" }] },
    { name: "Stay Out Long", futurePits: [{ lap: "48", durationMs: "24500" }] },
  ]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StrategyResponse | null>(null);

  const cutLapNum = Number(cutLap) || 1;

  function loadMonacoPreset() {
    setGrandPrix("Monaco Grand Prix");
    setDriverCode("VER");
    setCutLap("35");
    setHistory([
      { lap: "30", lapTimeMs: "75500" },
      { lap: "32", lapTimeMs: "76200" },
      { lap: "34", lapTimeMs: "76800" },
    ]);
    setScenarios([
      { name: "Aggressive Undercut", futurePits: [{ lap: "38", durationMs: "23500" }] },
      { name: "Conservative 1-Stop", futurePits: [{ lap: "40", durationMs: "24000" }] },
      { name: "Stay Out Long", futurePits: [{ lap: "48", durationMs: "24500" }] },
    ]);
    setResult(null);
    setError(null);
  }

  function loadAustralianPreset() {
    setGrandPrix("Australian Grand Prix");
    setDriverCode("HAM");
    setCutLap("25");
    setHistory([
      { lap: "18", lapTimeMs: "78000" },
      { lap: "21", lapTimeMs: "77500" },
      { lap: "24", lapTimeMs: "77000" },
    ]);
    setScenarios([
      { name: "Standard mid 1-stop", futurePits: [{ lap: "30", durationMs: "23000" }] },
      { name: "Super early 1-stop", futurePits: [{ lap: "22", durationMs: "23000" }] },
      { name: "Very late 1-stop", futurePits: [{ lap: "40", durationMs: "23000" }] },
      {
        name: "Attack 2-stop", futurePits: [
          { lap: "22", durationMs: "23000" },
          { lap: "40", durationMs: "23000" }
        ]
      },
    ]);
    setResult(null);
    setError(null);
  }

  function loadSingaporePreset() {
    setGrandPrix("Singapore Grand Prix");
    setDriverCode("VER");
    setCutLap("25");
    setHistory([
      { lap: "20", lapTimeMs: "105000" },
      { lap: "22", lapTimeMs: "106500" },
      { lap: "24", lapTimeMs: "108000" },
    ]);
    setScenarios([
      { name: "Emergency Stop Now", futurePits: [{ lap: "27", durationMs: "24000" }] },
      { name: "Push Through Pain", futurePits: [{ lap: "35", durationMs: "24000" }] },
      {
        name: "Balanced 2-Stop", futurePits: [
          { lap: "28", durationMs: "24000" },
          { lap: "45", durationMs: "24000" }
        ]
      },
      { name: "Gamble on Safety Car", futurePits: [{ lap: "40", durationMs: "24000" }] },
    ]);
    setResult(null);
    setError(null);
  }

  function clearAll() {
    setGrandPrix("Australian Grand Prix");
    setDriverCode("HAM");
    setCutLap("25");
    setHistory([]);
    setScenarios([{ name: "Scenario 1", futurePits: [] }]);
    setResult(null);
    setError(null);
  }

  function updateHistoryRow(index: number, field: keyof HistoryLap, value: string) {
    setHistory((prev) =>
      prev.map((row, i) => (i === index ? { ...row, [field]: value } : row)),
    );
  }

  function addHistoryRow() {
    setHistory((prev) => [...prev, { lap: "", lapTimeMs: "" }]);
  }

  function removeHistoryRow(index: number) {
    setHistory((prev) => prev.filter((_, i) => i !== index));
  }

  function updateScenarioName(index: number, value: string) {
    setScenarios((prev) =>
      prev.map((s, i) => (i === index ? { ...s, name: value } : s)),
    );
  }

  function addScenario() {
    setScenarios((prev) => [
      ...prev,
      { name: `Scenario ${prev.length + 1}`, futurePits: [] },
    ]);
  }

  function removeScenario(index: number) {
    setScenarios((prev) => prev.filter((_, i) => i !== index));
  }

  function updatePit(
    scenIdx: number,
    pitIdx: number,
    field: keyof FuturePit,
    value: string,
  ) {
    setScenarios((prev) =>
      prev.map((s, i) =>
        i === scenIdx
          ? {
            ...s,
            futurePits: s.futurePits.map((p, j) =>
              j === pitIdx ? { ...p, [field]: value } : p,
            ),
          }
          : s,
      ),
    );
  }

  function addPit(scenIdx: number) {
    setScenarios((prev) =>
      prev.map((s, i) =>
        i === scenIdx
          ? { ...s, futurePits: [...s.futurePits, { lap: "", durationMs: "" }] }
          : s,
      ),
    );
  }

  function removePit(scenIdx: number, pitIdx: number) {
    setScenarios((prev) =>
      prev.map((s, i) =>
        i === scenIdx
          ? { ...s, futurePits: s.futurePits.filter((_, j) => j !== pitIdx) }
          : s,
      ),
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const cut = Number(cutLap);
    if (!grandPrix.trim() || !driverCode.trim() || !Number.isFinite(cut)) {
      setError("Fill Grand Prix, driver code and a valid cut lap.");
      return;
    }

    const historyClean = history
      .filter((h) => h.lap && h.lapTimeMs)
      .map((h) => ({
        lap: Number(h.lap),
        lapTimeMs: Number(h.lapTimeMs),
      }));

    const scenariosClean = scenarios.map((s) => ({
      name: s.name || "Unnamed strategy",
      futurePits: s.futurePits
        .filter((p) => p.lap && p.durationMs)
        .map((p) => ({
          lap: Number(p.lap),
          durationMs: Number(p.durationMs),
        })),
    }));

    setLoading(true);
    try {
      const res = await fetch("https://captivating-emotion-production.up.railway.app/strategy/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grandPrix,
          driverCode,
          cutLap: cut,
          history: historyClean,
          scenarios: scenariosClean,
        }),
      });


      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend error (${res.status}): ${text}`);
      }

      const data = (await res.json()) as StrategyResponse;
      setResult(data);
    } catch (err: any) {
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function formatProb(p: number | undefined | null) {
    if (p == null) return "-";
    const clamped = Math.max(0, Math.min(1, p));
    return `${(clamped * 100).toFixed(1)}%`;
  }

  function ratingColor(rating?: number | null) {
    if (rating == null) return "bg-slate-500";
    if (rating >= 80) return "bg-emerald-500";
    if (rating >= 60) return "bg-sky-500";
    if (rating >= 40) return "bg-amber-500";
    return "bg-rose-500";
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4 md:p-8">
      <div className="mx-auto max-w-7xl">
        <Card className="border-slate-700/50 bg-slate-900/90 shadow-2xl backdrop-blur-sm">
          <CardHeader className="border-b border-slate-700/50 pb-6">
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="text-2xl font-bold text-slate-50 md:text-3xl">
                    F1 Strategy Advisor
                  </CardTitle>
                  <p className="mt-2 text-sm text-slate-400">
                    Compare pit strategies using LSTM model predictions
                  </p>
                </div>
                {result && (
                  <Badge className="w-fit bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-900">
                    Best: {result.bestScenario.name}
                  </Badge>
                )}
              </div>

              {/* Preset buttons */}
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="border-cyan-500/50 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20"
                  onClick={loadMonacoPreset}
                >
                  📍 Monaco Example
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="border-blue-500/50 bg-blue-500/10 text-blue-300 hover:bg-blue-500/20"
                  onClick={loadAustralianPreset}
                >
                  🇦🇺 Australian GP Example
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="border-purple-500/50 bg-purple-500/10 text-purple-300 hover:bg-purple-500/20"
                  onClick={loadSingaporePreset}
                >
                  🇸🇬 Singapore GP Example
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="border-slate-600 text-slate-400 hover:bg-slate-800"
                  onClick={clearAll}
                >
                  🗑️ Clear All
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-6">
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Race Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-slate-100">Race Configuration</h3>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="grandPrix" className="text-slate-200">Grand Prix</Label>
                    <Select value={grandPrix} onValueChange={setGrandPrix}>
                      <SelectTrigger id="grandPrix" className="bg-slate-800/50 border-slate-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="max-h-64">
                        {GRAND_PRIX_OPTIONS.map((gp) => (
                          <SelectItem key={gp.value} value={gp.value}>
                            {gp.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="driverCode" className="text-slate-200">Driver</Label>
                    <Select value={driverCode} onValueChange={setDriverCode}>
                      <SelectTrigger id="driverCode" className="bg-slate-800/50 border-slate-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {DRIVER_OPTIONS.map((d) => (
                          <SelectItem key={d.value} value={d.value}>
                            {d.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="cutLap" className="text-slate-200">Cut Lap: {cutLapNum}</Label>
                    <Slider
                      id="cutLap"
                      min={1}
                      max={70}
                      step={1}
                      value={[cutLapNum]}
                      onValueChange={(vals) => setCutLap(String(vals[0] ?? 1))}
                      className="pt-2"
                    />
                  </div>
                </div>
              </div>

              <Separator className="bg-slate-700/50" />

              {/* Two Column Layout */}
              <div className="grid gap-8 lg:grid-cols-2">
                {/* Left Column - History */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-100">History Laps</h3>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="border-slate-600 text-slate-200"
                      onClick={addHistoryRow}
                    >
                      + Add Lap
                    </Button>
                  </div>
                  <div className="space-y-3 max-h-80 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-800/30 p-4">
                    {history.length === 0 && (
                      <p className="text-sm text-slate-500">
                        Add recent lap times to improve model accuracy
                      </p>
                    )}
                    {history.map((row, i) => (
                      <div key={i} className="flex gap-3">
                        <div className="flex-1 space-y-1">
                          <Label className="text-xs text-slate-400">Lap</Label>
                          <Input
                            type="number"
                            value={row.lap}
                            onChange={(e) => updateHistoryRow(i, "lap", e.target.value)}
                            className="bg-slate-800 border-slate-700"
                          />
                        </div>
                        <div className="flex-1 space-y-1">
                          <Label className="text-xs text-slate-400">Time (ms)</Label>
                          <Input
                            type="number"
                            value={row.lapTimeMs}
                            onChange={(e) => updateHistoryRow(i, "lapTimeMs", e.target.value)}
                            className="bg-slate-800 border-slate-700"
                          />
                        </div>
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          className="mt-6 text-slate-400 hover:text-rose-400"
                          onClick={() => removeHistoryRow(i)}
                        >
                          ✕
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right Column - Scenarios */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-100">Scenarios</h3>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="border-slate-600 text-slate-200"
                      onClick={addScenario}
                    >
                      + Add Scenario
                    </Button>
                  </div>
                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {scenarios.map((s, scenIdx) => (
                      <div
                        key={scenIdx}
                        className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-4 space-y-3"
                      >
                        <div className="flex items-center gap-3">
                          <Input
                            value={s.name}
                            onChange={(e) => updateScenarioName(scenIdx, e.target.value)}
                            className="bg-slate-800 border-slate-700 font-medium"
                          />
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="text-slate-400 hover:text-rose-400"
                            onClick={() => removeScenario(scenIdx)}
                          >
                            ✕
                          </Button>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-slate-400">Pit Stops</span>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs text-slate-400"
                            onClick={() => addPit(scenIdx)}
                          >
                            + Add Stop
                          </Button>
                        </div>

                        {s.futurePits.length === 0 ? (
                          <p className="text-xs text-slate-500">No pit stops planned</p>
                        ) : (
                          <div className="space-y-2">
                            {s.futurePits.map((p, pitIdx) => (
                              <div key={pitIdx} className="flex gap-2">
                                <div className="flex-1 space-y-1">
                                  <Label className="text-[10px] text-slate-500">Lap</Label>
                                  <Input
                                    type="number"
                                    value={p.lap}
                                    onChange={(e) => updatePit(scenIdx, pitIdx, "lap", e.target.value)}
                                    className="h-9 bg-slate-900 border-slate-700 text-sm"
                                  />
                                </div>
                                <div className="flex-1 space-y-1">
                                  <Label className="text-[10px] text-slate-500">Duration (ms)</Label>
                                  <Input
                                    type="number"
                                    value={p.durationMs}
                                    onChange={(e) => updatePit(scenIdx, pitIdx, "durationMs", e.target.value)}
                                    className="h-9 bg-slate-900 border-slate-700 text-sm"
                                  />
                                </div>
                                <Button
                                  type="button"
                                  size="icon"
                                  variant="ghost"
                                  className="mt-5 h-9 w-9 text-slate-500 hover:text-rose-400"
                                  onClick={() => removePit(scenIdx, pitIdx)}
                                >
                                  ✕
                                </Button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-cyan-500 py-6 text-lg font-semibold text-slate-900 hover:bg-cyan-400"
              >
                {loading ? "Analyzing Strategies..." : "Score Strategies"}
              </Button>

              {error && (
                <div className="rounded-lg border border-rose-500/50 bg-rose-500/10 p-4 text-sm text-rose-200">
                  {error}
                </div>
              )}

              {/* Results */}
              {result && (
                <>
                  <Separator className="bg-slate-700/50" />
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-slate-100">Model Verdict</h3>
                    <div className="grid gap-4">
                      {result.allScenarios.map((sc, idx) => (
                        <div
                          key={idx}
                          className={`rounded-lg border p-4 transition-all ${sc.name === result.bestScenario.name
                              ? "border-emerald-500/70 bg-emerald-500/10 shadow-lg shadow-emerald-500/20"
                              : "border-slate-700/50 bg-slate-800/30"
                            }`}
                        >
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <h4 className="text-lg font-semibold text-slate-50">{sc.name}</h4>
                            <div className="flex items-center gap-3">
                              <Badge className={`${ratingColor(sc.rating ?? 0)} px-3 py-1 text-sm font-bold text-slate-900`}>
                                {sc.rating ?? 0}/100
                              </Badge>
                              <span className="text-sm text-slate-300">
                                {formatProb(sc.successProb)} success
                              </span>
                            </div>
                          </div>
                          <Progress value={(sc.rating ?? 0) as number} className="mt-3 h-2 bg-slate-700" />
                          <p className="mt-3 text-sm text-slate-400">
                            {sc.futurePits.length === 0
                              ? "Runs to the flag with no further stops"
                              : `${sc.futurePits.length} stop${sc.futurePits.length > 1 ? "s" : ""} at lap${sc.futurePits.length > 1 ? "s" : ""} ${sc.futurePits.map((p) => p.lap).join(", ")}`}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
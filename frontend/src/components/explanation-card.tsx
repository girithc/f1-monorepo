import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';

type FeatureImpact = {
  name: string;
  impact: number;
  direction?: string;
};

interface ExplanationCardProps {
  featureImpacts: FeatureImpact[];
}

function friendlyName(name: string) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function ExplanationCard({ featureImpacts }: ExplanationCardProps) {
  const [showPositive, setShowPositive] = useState(true);

  const maxAbs = Math.max(...featureImpacts.map((f) => Math.abs(f.impact)), 0.001);

  const filtered = useMemo(() => {
    return featureImpacts.filter((f) => (showPositive ? f.impact > 0 : f.impact < 0));
  }, [featureImpacts, showPositive]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Model Explanation</CardTitle>
            <CardDescription>Top feature impacts on predicted finish</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Better</span>
            <Switch checked={showPositive} onCheckedChange={(v) => setShowPositive(Boolean(v))} />
            <span className="text-xs text-muted-foreground">Worse</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {featureImpacts.length === 0 ? (
          <div className="text-sm text-muted-foreground">No explanation available for this prediction.</div>
        ) : (
          <ul className="space-y-3">
            {filtered.length === 0 && (
              <li className="text-sm text-muted-foreground">No features match the selected sign.</li>
            )}
            {filtered.map((f) => (
              <li key={f.name}>
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">{friendlyName(f.name)}</div>
                  <div className="text-sm text-muted-foreground">{f.impact.toFixed(3)}</div>
                </div>

                <div className="mt-2">
                  <div className="h-2 w-full rounded bg-muted">
                    <div
                      className={`h-2 rounded ${f.impact < 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                      style={{ width: `${(Math.abs(f.impact) / maxAbs) * 100}%` }}
                    />
                  </div>
                  {f.direction && (
                    <div className="mt-1 text-xs text-muted-foreground">{f.direction}</div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

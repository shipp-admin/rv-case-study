# Validation UI Setup Guide

## Overview

Interactive dashboard with **Run & Validate** buttons for each analysis subphase, integrated with the existing Next.js application.

## Architecture Summary

### Two Modes:
1. **Development Mode**: Run Python analysis scripts, validate outputs, view results
2. **Results Mode**: View finalized analysis organized by research questions (Q1, Q2, Q3)

## Directory Structure Created

```
app/
├── dashboard/                    # ✅ Created
│   ├── page.tsx                  # TODO: Main dashboard with mode toggle
│   ├── layout.tsx                # TODO: Dashboard layout with sidebar
│   └── components/               # ✅ Created
│       ├── ValidationSidebar.tsx      # TODO: Phase/subphase run buttons
│       ├── ConsoleOutput.tsx          # TODO: Real-time Python output
│       ├── ValidationResults.tsx      # TODO: Validation results display
│       ├── ModeToggle.tsx             # TODO: Dev ⇄ Results toggle
│       ├── ProgressTracker.tsx        # TODO: Phase completion status
│       ├── question1/            # ✅ Created (for Results Mode)
│       ├── question2/            # ✅ Created (for Results Mode)
│       └── question3/            # ✅ Created (for Results Mode)
│
├── api/                          # API routes for Python execution
│   ├── analysis/run/             # ✅ Created
│   │   └── route.ts              # TODO: POST /api/analysis/run
│   ├── validation/run/           # ✅ Created
│   │   └── route.ts              # TODO: POST /api/validation/run
│   └── outputs/list/             # ✅ Created
│       └── route.ts              # TODO: GET /api/outputs/list
│
└── lib/                          # ✅ Created
    ├── python-executor.ts        # TODO: Execute Python scripts
    ├── validation-parser.ts      # TODO: Parse JSON output
    └── types.ts                  # TODO: TypeScript interfaces
```

## Implementation Steps

### Step 1: Update Python Scripts (Phase 1.1 first)

**File**: `src/phase1_eda/univariate.py`

Add JSON output at the end:
```python
import json
import time

def generate_univariate_summary(df, output_dir='reports/phase1_eda'):
    start_time = time.time()

    # ... existing analysis code ...

    # Build structured output for UI
    output = {
        "success": True,
        "subphase": "Phase 1.1: Univariate Analysis",
        "insights": [
            f"FICO 800+: 45.8% approval vs <580: 2.8%",
            f"Lender C most lenient (17.1% approval)",
            f"Employment: 12.1% vs 5.5% (2.2x difference)"
        ],
        "outputs": {
            "tables": ["approval_by_reason.csv", ...],
            "figures": ["approval_by_fico_bins.png", ...]
        },
        "execution_time": time.time() - start_time
    }

    # Output JSON marker for UI parsing
    print("\n__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))

    return results
```

**File**: `tests/test_phase1_univariate.py`

Add JSON output at the end:
```python
import json

def validate_phase1_univariate():
    # ... existing validation code ...

    validation_output = {
        "success": pass_rate >= 0.80,
        "subphase": "Phase 1.1: Univariate Analysis",
        "checks_passed": passed_checks,
        "total_checks": total_checks,
        "pass_rate": pass_rate,
        "details": [
            {
                "category": "Tables",
                "checks": [
                    {"name": "approval_by_reason.csv", "passed": True, "message": "✅ Found"}
                ]
            }
        ],
        "insights": [
            "FICO 800+: 45.8% approval vs <580: 2.8%",
            "Lender C most lenient (17.1% approval)"
        ]
    }

    print("\n__JSON_OUTPUT__")
    print(json.dumps(validation_output, indent=2))

    return validation_output["success"]
```

### Step 2: Create API Route for Running Analysis

**File**: `app/api/analysis/run/route.ts`

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  const { phase, subphase, script } = await request.json();

  try {
    const { stdout, stderr } = await execAsync(`python3 ${script}`);

    // Parse JSON output
    const jsonMatch = stdout.match(/__JSON_OUTPUT__\n([\s\S]*)/);
    const result = jsonMatch ? JSON.parse(jsonMatch[1]) : null;

    return NextResponse.json({
      success: true,
      logs: stdout.split('\n'),
      result: result,
      executionTime: result?.execution_time || 0
    });
  } catch (error: any) {
    return NextResponse.json({
      success: false,
      error: error.message,
      logs: error.stdout ? error.stdout.split('\n') : []
    }, { status: 500 });
  }
}
```

### Step 3: Create API Route for Running Validation

**File**: `app/api/validation/run/route.ts`

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  const { test } = await request.json();

  try {
    const { stdout, stderr } = await execAsync(`python3 ${test}`);

    // Parse JSON output
    const jsonMatch = stdout.match(/__JSON_OUTPUT__\n([\s\S]*)/);
    const result = jsonMatch ? JSON.parse(jsonMatch[1]) : null;

    return NextResponse.json({
      success: result?.success || false,
      checks_passed: result?.checks_passed || 0,
      total_checks: result?.total_checks || 0,
      pass_rate: result?.pass_rate || 0,
      details: result?.details || [],
      insights: result?.insights || []
    });
  } catch (error: any) {
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}
```

### Step 4: Create ValidationSidebar Component

**File**: `app/dashboard/components/ValidationSidebar.tsx`

```typescript
'use client';

import { useState } from 'react';

interface SubphaseStatus {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  script: string;
  test: string;
}

export default function ValidationSidebar() {
  const [subphases, setSubphases] = useState<SubphaseStatus[]>([
    {
      id: 'phase1_1',
      name: '1.1 Univariate',
      status: 'pending',
      script: 'src/phase1_eda/univariate.py',
      test: 'tests/test_phase1_univariate.py'
    },
    // ... more subphases
  ]);

  const runAnalysis = async (subphase: SubphaseStatus) => {
    setSubphases(prev => prev.map(s =>
      s.id === subphase.id ? { ...s, status: 'running' } : s
    ));

    const res = await fetch('/api/analysis/run', {
      method: 'POST',
      body: JSON.stringify({
        phase: 'phase1_eda',
        subphase: 'univariate',
        script: subphase.script
      })
    });

    const result = await res.json();

    setSubphases(prev => prev.map(s =>
      s.id === subphase.id ? {
        ...s,
        status: result.success ? 'passed' : 'failed'
      } : s
    ));
  };

  const runValidation = async (subphase: SubphaseStatus) => {
    const res = await fetch('/api/validation/run', {
      method: 'POST',
      body: JSON.stringify({ test: subphase.test })
    });

    const result = await res.json();

    setSubphases(prev => prev.map(s =>
      s.id === subphase.id ? {
        ...s,
        status: result.success ? 'passed' : 'failed'
      } : s
    ));
  };

  return (
    <aside className="w-64 bg-gray-50 p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">Analysis Pipeline</h2>

      <div className="space-y-2">
        <div>
          <h3 className="font-semibold text-sm mb-2">Phase 1: EDA</h3>
          {subphases.filter(s => s.id.startsWith('phase1')).map(subphase => (
            <div key={subphase.id} className="mb-3 p-2 bg-white rounded shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">{subphase.name}</span>
                {subphase.status === 'passed' && <span>✅</span>}
                {subphase.status === 'failed' && <span>❌</span>}
                {subphase.status === 'running' && <span>🔄</span>}
                {subphase.status === 'pending' && <span>⏳</span>}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => runAnalysis(subphase)}
                  className="text-xs bg-blue-500 text-white px-2 py-1 rounded"
                  disabled={subphase.status === 'running'}
                >
                  ▶ Run
                </button>
                <button
                  onClick={() => runValidation(subphase)}
                  className="text-xs bg-green-500 text-white px-2 py-1 rounded"
                  disabled={subphase.status === 'running'}
                >
                  ✓ Validate
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
```

### Step 5: Create Dashboard Page

**File**: `app/dashboard/page.tsx`

```typescript
'use client';

import { useState } from 'react';
import ValidationSidebar from './components/ValidationSidebar';
import ConsoleOutput from './components/ConsoleOutput';
import ValidationResults from './components/ValidationResults';

export default function DashboardPage() {
  const [mode, setMode] = useState<'development' | 'results'>('development');

  return (
    <div className="flex h-screen">
      <ValidationSidebar />

      <main className="flex-1 p-6 overflow-y-auto">
        <div className="mb-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Lending Analysis Dashboard</h1>

          <button
            onClick={() => setMode(mode === 'development' ? 'results' : 'development')}
            className="px-4 py-2 bg-gray-200 rounded"
          >
            {mode === 'development' ? '📊 Results Mode' : '🔧 Development Mode'}
          </button>
        </div>

        {mode === 'development' ? (
          <div>
            <ConsoleOutput />
            <ValidationResults />
          </div>
        ) : (
          <div>
            {/* Results mode components go here */}
            <p>Results mode - Q1, Q2, Q3 visualizations</p>
          </div>
        )}
      </main>
    </div>
  );
}
```

## Next Steps

1. ✅ PRD updated with interactive validation UI
2. ✅ Directory structure created
3. **TODO**: Update `src/phase1_eda/univariate.py` with JSON output
4. **TODO**: Update `tests/test_phase1_univariate.py` with JSON output
5. **TODO**: Create API routes (`app/api/analysis/run/route.ts`, `app/api/validation/run/route.ts`)
6. **TODO**: Create UI components (ValidationSidebar, ConsoleOutput, ValidationResults)
7. **TODO**: Create dashboard page (`app/dashboard/page.tsx`)
8. **TODO**: Test Phase 1.1 with new UI

## Testing the New System

Once implemented, you can:
1. Navigate to `http://localhost:3000/dashboard`
2. Click "▶ Run" on Phase 1.1 to execute `univariate.py`
3. See real-time console output in the UI
4. Click "✓ Validate" to run validation tests
5. View validation results with pass/fail status
6. See key insights and generated files
7. Toggle to Results Mode to view Q1/Q2/Q3 visualizations

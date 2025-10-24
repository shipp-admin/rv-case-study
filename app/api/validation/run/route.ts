import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const { test } = await request.json();

    if (!test) {
      return NextResponse.json(
        { success: false, error: 'Test path is required' },
        { status: 400 }
      );
    }

    // Convert file path to module path (e.g., tests/test_phase1_bivariate.py -> tests.test_phase1_bivariate)
    const modulePath = test.replace(/\.py$/, '').replace(/\//g, '.');

    // Execute validation test as module to support relative imports
    const { stdout, stderr } = await execAsync(`python3 -m ${modulePath}`, {
      cwd: process.cwd(),
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
    });

    // Extract JSON output
    const lines = stdout.split('\n');
    let jsonStart = -1;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('__JSON_OUTPUT__')) {
        jsonStart = i + 1;
        break;
      }
    }

    let result = null;
    if (jsonStart !== -1) {
      const jsonLines: string[] = [];
      let braceCount = 0;
      for (let i = jsonStart; i < lines.length; i++) {
        jsonLines.push(lines[i]);
        braceCount += (lines[i].match(/{/g) || []).length;
        braceCount -= (lines[i].match(/}/g) || []).length;
        if (braceCount === 0 && jsonLines.join('\n').includes('{')) {
          break;
        }
      }
      try {
        result = JSON.parse(jsonLines.join('\n'));
      } catch (e) {
        console.error('Failed to parse JSON output:', e);
      }
    }

    return NextResponse.json({
      success: result?.success || false,
      checks_passed: result?.checks_passed || 0,
      total_checks: result?.total_checks || 0,
      pass_rate: result?.pass_rate || 0,
      details: result?.details || [],
      insights: result?.insights || [],
      logs: stdout.split('\n'),
      stderr: stderr || null,
    });
  } catch (error: any) {
    console.error('Validation test error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
        checks_passed: 0,
        total_checks: 0,
        pass_rate: 0,
        details: [],
        insights: [],
        logs: error.stdout ? error.stdout.split('\n') : [],
        stderr: error.stderr || null,
      },
      { status: 500 }
    );
  }
}

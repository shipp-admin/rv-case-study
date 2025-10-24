import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const filePath = join(process.cwd(), 'reports', ...params.path);
    const fileBuffer = await readFile(filePath);

    // Determine content type based on file extension
    const ext = params.path[params.path.length - 1].split('.').pop()?.toLowerCase();
    const contentType = ext === 'png' ? 'image/png'
                      : ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg'
                      : ext === 'csv' ? 'text/csv'
                      : 'application/octet-stream';

    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: 'File not found', message: error.message },
      { status: 404 }
    );
  }
}

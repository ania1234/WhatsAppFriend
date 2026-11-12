// Saves one rater's answers as one JSON file in Vercel Blob: ratings/<time>-<name>-<random>.json
import { put } from '@vercel/blob';

const MAX_BODY_BYTES = 200_000;
// Must match how the Blob store was created ("private" or "public").
const BLOB_ACCESS = process.env.BLOB_ACCESS || 'private';

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function isValidRating(rating) {
  return (
    rating !== null &&
    typeof rating === 'object' &&
    Number.isInteger(rating.question_id) &&
    typeof rating.best === 'string' &&
    typeof rating.worst === 'string' &&
    rating.best !== rating.worst &&
    typeof rating.passes === 'boolean' &&
    Array.isArray(rating.order_shown)
  );
}

export async function POST(request) {
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) {
    return json(413, { error: 'Submission too large.' });
  }

  let submission;
  try {
    submission = JSON.parse(text);
  } catch {
    return json(400, { error: 'Body is not valid JSON.' });
  }

  const rater = typeof submission?.rater === 'string' ? submission.rater.trim().slice(0, 60) : '';
  const ratings = submission?.ratings;
  if (!rater || !Array.isArray(ratings) || ratings.length === 0 || !ratings.every(isValidRating)) {
    return json(400, { error: 'Submission has the wrong shape.' });
  }

  const receivedAt = new Date().toISOString();
  const record = {
    rater,
    started_at: submission.started_at ?? null,
    finished_at: submission.finished_at ?? null,
    received_at: receivedAt,
    ratings: ratings.map((rating) => ({
      question_id: rating.question_id,
      best: rating.best,
      worst: rating.worst,
      passes: rating.passes,
      order_shown: rating.order_shown.map(String).slice(0, 20),
    })),
  };

  const slug = rater.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'rater';
  const pathname = `ratings/${receivedAt.replace(/[:.]/g, '-')}-${slug}.json`;

  try {
    await put(pathname, JSON.stringify(record, null, 2), {
      access: BLOB_ACCESS,
      addRandomSuffix: true,
      contentType: 'application/json',
    });
  } catch (error) {
    console.error('Blob upload failed:', error);
    return json(500, { error: 'Could not save the submission.' });
  }

  return json(200, { ok: true });
}

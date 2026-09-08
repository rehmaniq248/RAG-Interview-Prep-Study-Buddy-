# Example: Project Write-Up

> This is a sample file so you can test the tool before adding your own
> documents. It describes a fictional project. Replace it with your real
> write-ups — the tool is only as useful as what you put in this folder.

## Project: Trailhead — a route-planning app for day hikers

### The problem
Hikers planning a day trip had to cross-reference three separate sources:
trail conditions from a park service RSS feed, elevation profiles from a
topographic API, and weather forecasts. There was no single view that answered
the actual question: "can I finish this trail before dark, given today's
conditions?" I built Trailhead to answer that in one screen.

### My role
Solo project over about four months. I designed the data model, wrote the
backend and the frontend, and deployed it. Roughly 6,000 lines of Python and
TypeScript.

### Architecture and the tradeoffs
The backend is a FastAPI service backed by PostgreSQL with the PostGIS
extension. I chose PostGIS over storing raw coordinate arrays because the core
query — "find trails whose path intersects this bounding box" — is a spatial
one, and doing that in application code meant loading every trail into memory
on each request. PostGIS made it an indexed query that returned in under 40ms
against 12,000 trails.

The biggest tradeoff was caching. Trail conditions change slowly (the park
service updates roughly daily) but weather changes hourly, and I was calling
both on every page load. I split the cache: conditions cached for 6 hours,
weather for 20 minutes, both in Redis. That cut external API calls by about 85%
and brought median page load from 2.1s to 340ms. The cost was staleness — a
trail closure could take up to 6 hours to appear. I decided that was acceptable
for a planning tool used the night before a hike, but it would have been the
wrong call for a live safety app.

### What went wrong
My first version computed the "can you finish before dark" estimate using a
flat average hiking speed of 3 mph. It was badly wrong on steep trails — it
told users they could finish a 4,000-foot climb in half the time it actually
takes. I replaced it with Naismith's rule plus a correction for descent, which
adds time proportional to elevation gain. I only caught this because a friend
tested it on a real trail and came back after dark. The lesson I took from it
was that I had validated the code but never validated the model against
reality, and no amount of unit testing would have caught it.

### Outcome
About 400 monthly active users at its peak, mostly from a single Reddit post.
I shut it down after a year because the topographic API moved to a paid tier
that I couldn't justify for a free side project.

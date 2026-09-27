export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "https://orient78.github.io",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: cors });
    }

    if (request.method !== "POST") {
      return Response.json(
        { ok: false, message: "POST only" },
        { status: 405, headers: cors }
      );
    }

    try {
      // 1. Check environment variables
      if (!env.REFRESH_PIN) {
        return Response.json(
          { ok: false, message: "REFRESH_PIN secret is missing" },
          { status: 500, headers: cors }
        );
      }

      if (!env.GITHUB_TOKEN) {
        return Response.json(
          { ok: false, message: "GITHUB_TOKEN secret is missing" },
          { status: 500, headers: cors }
        );
      }

      // 2. Read request body
      const body = await request.json();

      if (!body.pin) {
        return Response.json(
          { ok: false, message: "PIN is missing" },
          { status: 400, headers: cors }
        );
      }

      if (body.pin !== env.REFRESH_PIN) {
        return Response.json(
          { ok: false, message: "PIN incorrect" },
          { status: 401, headers: cors }
        );
      }

      // 3. Trigger GitHub Actions
      const response = await fetch(
        "https://api.github.com/repos/orient78/taiwan-stock-scanner/actions/workflows/daily_scan.yml/dispatches",
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "taiwan-stock-refresh",
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            ref: "main"
          })
        }
      );

      // GitHub workflow_dispatch success = HTTP 204
      if (!response.ok) {
        const githubError = await response.text();

        return Response.json(
          {
            ok: false,
            message: "GitHub trigger failed",
            status: response.status,
            github_error: githubError
          },
          { status: 502, headers: cors }
        );
      }

      return Response.json(
        {
          ok: true,
          message: "Stock scan started",
          github_status: response.status
        },
        { status: 200, headers: cors }
      );

    } catch (error) {
      return Response.json(
        {
          ok: false,
          message: "Worker error",
          error: String(error)
        },
        { status: 500, headers: cors }
      );
    }
  }
};

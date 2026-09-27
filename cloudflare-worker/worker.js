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
      const body = await request.json();

      if (body.pin !== env.REFRESH_PIN) {
        return Response.json(
          { ok: false, message: "PIN incorrect" },
          { status: 401, headers: cors }
        );
      }

      const response = await fetch(
        "https://api.github.com/repos/orient78/taiwan-stock-scanner/actions/workflows/daily_scan.yml/dispatches",
        {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "taiwan-stock-refresh"
          },
          body: JSON.stringify({
            ref: "main"
          })
        }
      );

      if (!response.ok) {
        return Response.json(
          {
            ok: false,
            message: "GitHub trigger failed",
            status: response.status
          },
          { status: 502, headers: cors }
        );
      }

      return Response.json(
        {
          ok: true,
          message: "Stock scan started"
        },
        { headers: cors }
      );

    } catch (error) {
      return Response.json(
        { ok: false, message: "Worker error" },
        { status: 500, headers: cors }
      );
    }
  }
};

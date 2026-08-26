# HTTPYac API tests

These `.http` requests exercise the running Grinder Diagnostics Model API. Pytest ignores them
because pytest collection remains limited to Python test modules.

The npm registry is inherited from `~/.npmrc`:

`https://packagefeedproxy.microsoft.io/npm/`

Start the API:

```bash
mise run f:api:serve
```

Install the pinned HTTPYac dependency once:

```bash
mise run l:http:setup
```

Run every request and assertion:

```bash
mise run m:http:test
```

Editors with an HTTPYac extension can execute individual requests directly from:

- `health.http`
- `predict.http`
- `validation.http`

Override `baseUrl` from the editor or CLI when testing another endpoint.

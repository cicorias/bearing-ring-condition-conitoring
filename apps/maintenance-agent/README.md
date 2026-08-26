# Maintenance agent application

This directory is reserved for the future Microsoft Agent Framework application.

The application will consume the typed response from the local Grinder Diagnostics Model API. It
will not receive raw TDMS signals or run model training. The intended responsibilities are:

- stop silently for healthy cycles;
- enrich detected faults with maintenance and cost context;
- coordinate bounded agent reasoning;
- require human approval before state-changing actions;
- emit traceable downstream recommendations.

No agent runtime is implemented yet. The local model/API remains independently runnable.

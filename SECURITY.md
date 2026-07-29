# Security Policy

FPS is a scientific/offline command-line and desktop tool (RVE geometry
generation); it does not run as a network service and does not process
untrusted network input. The main realistic risks are in file parsing
(e.g. malformed config/mesh files) and in the optional PyQt6 GUI.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security concern.
Instead, email exauce-devine.ngouloubi@unicaen.fr with:

- A description of the issue and its potential impact
- Steps to reproduce, or a minimal example file/script
- The FPS version / commit hash affected

You should get an acknowledgement within a few business days. Once a fix is
available, a new release will be tagged and the reporter credited (unless
anonymity is requested).

## Supported versions

Only the latest tagged release and the `main` branch are supported.

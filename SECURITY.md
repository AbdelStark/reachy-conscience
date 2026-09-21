# Security

This guard is defense in depth, not a certified safety system. Physical emergency stop, SDK motor limits, access control, and owner supervision are independent requirements. An app must call the guard before execution, fail closed on errors, and avoid monkeypatching an already-executing action path. Untrusted text must remain data, never instructions. Do not log secrets, transcripts, or images. Report vulnerabilities through GitHub's private vulnerability reporting feature.

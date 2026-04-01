"""
SES setup for HumanGuard — run once to verify the sender address and
print production-access request instructions.

Usage:
    python infrastructure/ses_setup.py [--sender noreply@humanguard.net]

This script:
1. Verifies the sender email address with AWS SES (sends a confirmation email
   to that address — you must click the link before SES will send from it).
2. Prints instructions for requesting production access (moving out of sandbox).
"""

import argparse
import sys


REGION = "us-east-1"
DEFAULT_SENDER = "noreply@humanguard.net"


def verify_sender(sender: str, region: str = REGION) -> None:
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is not installed. Run: pip install boto3")
        sys.exit(1)

    ses = boto3.client("ses", region_name=region)

    # Check if already verified
    resp = ses.list_verified_email_addresses()
    if sender in resp.get("VerifiedEmailAddresses", []):
        print(f"✓ {sender} is already verified in SES ({region}).")
        return

    ses.verify_email_identity(EmailAddress=sender)
    print(f"Verification email sent to: {sender}")
    print("Check your inbox and click the verification link before emails will be sent.")


def print_production_instructions() -> None:
    print(
        """
────────────────────────────────────────────────────────────────
  Requesting SES Production Access (exit sandbox)
────────────────────────────────────────────────────────────────

By default AWS SES operates in sandbox mode — you can only send
to verified email addresses.  To send to arbitrary recipients:

1. Open the AWS Console → SES → Account dashboard
   https://us-east-1.console.aws.amazon.com/ses/home?region=us-east-1

2. Click "Request production access" (or use the CLI):

   aws sesv2 put-account-details \\
       --mail-type TRANSACTIONAL \\
       --website-url https://humanguard.net \\
       --use-case-description "HumanGuard sends API-key verification emails to users who register for the bot-detection API. One email per registration. Users explicitly request the email by submitting the /api/register form." \\
       --region us-east-1

3. AWS reviews the request (typically within 24 hours).

4. Set the SENDER_EMAIL environment variable to the verified address:

   SENDER_EMAIL=noreply@humanguard.net

   In Lambda this is set automatically by scripts/aws_deploy.sh.
────────────────────────────────────────────────────────────────
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up AWS SES for HumanGuard")
    parser.add_argument(
        "--sender",
        default=DEFAULT_SENDER,
        help=f"Sender email address to verify (default: {DEFAULT_SENDER})",
    )
    parser.add_argument(
        "--region",
        default=REGION,
        help=f"AWS region (default: {REGION})",
    )
    args = parser.parse_args()

    print(f"Setting up SES in {args.region} for sender: {args.sender}\n")
    verify_sender(args.sender, args.region)
    print_production_instructions()


if __name__ == "__main__":
    main()

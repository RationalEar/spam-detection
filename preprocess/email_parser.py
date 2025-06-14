import email
from email import policy
import re


def extract_email_address(header_value):
    # Extracts the email address from a header value
    match = re.search(r'[\w.-]+@[\w.-]+', header_value or "")
    return match.group(0) if match else ""


def remove_server_signatures(body):
    # Removes common server signatures (e.g., lines starting with '-- ' or 'Sent from')
    # and trims trailing whitespace
    lines = body.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('-- '):
            break
        if re.match(r'^\s*Sent from', line):
            break
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()


def parse_email(file_path):
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        sender_email = extract_email_address(sender)
        sender_domain = sender_email.split('@')[-1] if sender_email else ""
        reply_to = msg.get("Reply-To", "")
        reply_to_email = extract_email_address(reply_to)
        date = msg.get("Date", "")  # Extract the Date header

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        # Remove server signatures from body
        body = remove_server_signatures(body)

        return {
            "subject": subject,
            "sender": sender,
            "sender_domain": sender_domain,
            "reply_to": reply_to_email,
            "body": body,
            "date": date  # Add date to output
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None
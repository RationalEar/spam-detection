import email
from email import policy


def parse_email(file_path):
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        subject = msg.get("Subject", "")
        sender = msg.get("From", "")
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        
        return {"subject": subject, "sender": sender, "body": body}
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None
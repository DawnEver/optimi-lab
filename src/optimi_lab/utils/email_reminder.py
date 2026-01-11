import resend

from optimi_lab.utils.config import CONF

resend.api_key = CONF.utils.resend_api_key


def email_reminder(user_email, subject, message):
    """Sends an email reminder to the specified user.

    Parameters
    ----------
    user_email (str): The email address of the user.
    subject (str): The subject of the email.
    message (str): The body of the email.

    """
    params: resend.Emails.SendParams = {
        'from': "Mingyang's Robot<no-reply@mingyangbao.site>",
        'to': [user_email],
        'subject': subject,
        'html': message,
    }
    resend.Emails.send(params)


if __name__ == '__main__':
    email_reminder('mingyangbob@gmail.com', 'Test Subject', '<h1>This is a test email</h1>')

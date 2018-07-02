
"""
Lists the user's Gmail labels.
"""
from apiclient.discovery import build
import base64
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from httplib2 import Http
import mimetypes
from oauth2client import file, client, tools





def mail(message_text = 'default', sender= 'bentummartijn@gmail.com', to='bentummartijn@gmail.com', subject = 'default',send = True,attach= ''):
	"""Create a message for an email.

	message_text 		the body of the email
	sender 				email address of the sender 
	to 					email address of the recipient
	subject 			subject of the email
	send 				whether to send the email
	attach 				filename or filenames (comma seperated) of the attachment

	Returns:
	An object containing a base64url encoded email object.
	"""
	t = MIMEText(message_text)
	message = MIMEMultipart()
	message['to'] = to
	message['from'] = sender
	message['subject'] = subject
	# message.preamble = message_text
	message.attach(t)
	if attach:
		for attachment in attach.split(','):
			content_type,encoding = mimetypes.guess_type(attach)
			main_type, sub_type = content_type.split('/', 1)
			fp = open(attachment,'rb')
			pic = MIMEImage(fp.read(), _subtype=sub_type)
			message.attach(pic)
	message = {'raw': base64.urlsafe_b64encode(message.as_bytes())}
	message['raw'] = message['raw'].decode()  
	
	if send:
		service = get_service()
		message = send_message(message,service,user_id=sender)
	return message

def get_service():
	'''Loads the credentials to interact with the gmail api.'''
	store = file.Storage('MAIL/send_credentials.json')
	creds = store.get()
	service = build('gmail', 'v1', http=creds.authorize(Http()))
	return service


def send_message(message='default',service=None, user_id='me'):
	"""Send an email message.

	service: 		Authorized Gmail API service instance.
	user_id: 		User's email address. The special value "me"
					can be used to indicate the authenticated user.
	message: 		Message to be sent.

	Returns:
	Sent Message.
	"""
	if service == None: service = get_service()
	try:
		message = (service.users().messages().send(userId=user_id, body=message).execute())
		print('Message Id: %s' % message['id'])
		return message
	except: #errors.HttpError, error:
		print('An error occurred:')# %s' % error)


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
import email
from apiclient import errors
import time


def get_service():
	'''Loads the credentials to interact with the gmail api.'''
	store = file.Storage('MAIL/listener_credentials.json')
	creds = store.get()
	service = build('gmail', 'v1', http=creds.authorize(Http()))
	return service


def get_d_t_epoch(payload):
	try:
		d =' '.join(payload['headers'][-3]['value'].split(' ')[1:-1])
		t = time.strptime(d,'%d %b %Y %H:%M:%S')
		epoch = time.mktime(t)
		return d, t, epoch
	except: return False, False, False


def get_last_mail(query = 'setpercnow'):
	mails = get_mails(query)
	if not mails: return False
	if len(mails) == 1: return mails[0]
	latest = 0
	latest_index = 0
	found = False
	for i,m in enumerate(mails):
		payload = m['payload']
		d,t,epoch = get_d_t_epoch(payload)
		if not d: continue
		if epoch > latest: 
			found = True
			latest = epoch
			latest_index = i
			print('current latest mail:',m['snippet'],'at:',d)
	return mails[latest_index]
	

def get_mails(query ='setpercnow'):
	service = get_service()
	messages = query_message(service,'me',query)
	print('found messages:',messages)
	if len(messages) > 0 and type(messages[0]) == dict and 'id' in messages[0].keys():
		return [get_message(service,'me',m['id']) for m in messages]
	else: 
		print('found no mails')
		return False


def get_message(service, user_id, msg_id):
	"""Get a Message with given ID.
	service: 	Authorized Gmail API service instance.
	user_id: 	User's email address. The special value "me"
				can be used to indicate the authenticated user.
	msg_id: 	the ID of the Message required.

	Returns:
	A Message.
	"""
	try:
		message = service.users().messages().get(userId=user_id, id=msg_id).execute()
		# print 'Message snippet: %s' % message['snippet']
		return message
	except:
		print('An error occurred')


def query_message(service, user_id, query=''):
	"""List all Messages of the user's mailbox matching the query.
	service: 	Authorized Gmail API service instance.
	user_id: 	User's email address. The special value "me"
				can be used to indicate the authenticated user.
	query: 		String used to filter messages returned.
				Eg.- 'from:user@some_domain.com' for Messages from a particular sender.

	Returns:
	List of Messages that match the criteria of the query. Note that the
	returned list contains Message IDs, you must use get with the
	appropriate ID to get the details of a Message.
	"""

	try:
		response = service.users().messages().list(userId=user_id,
											   q=query).execute()
		messages = []
		if 'messages' in response:
			messages.extend(response['messages'])
		while 'nextPageToken' in response:
			page_token = response['nextPageToken']
			response = service.users().messages().list(userId=user_id, q=query,
											 pageToken=page_token).execute()
			messages.extend(response['messages'])
		return messages
	except:
		print('An error occurred')




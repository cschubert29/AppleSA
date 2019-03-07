from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

consumer_key = "XXX"
consumer_secret = "XXX"
access_token = "XXX"
access_token_secret = "XXX"

class APIListener(StreamListener):

    def on_data(self, data):
        print data
        tweetdata = open("FILENAME", "a")
        tweetdata.write(data +"\n")
        tweetdata.close()

        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles the connection to Twitter Streaming API
    listen = APIListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listen)

    #This line filters Twitter Streams
    stream.filter(languages=['en'], track=['@apple', '#apple', '#AAPL', '#iphone', '#Iphone', '#Ipad', '#ipad',
                                           '#mac', '#Mac', '#appleTV', '#appletv', '#AppleTV', '#iwatch', '#Iwatch'])
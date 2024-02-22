USER_AVATAR_URL = "https://t4.ftcdn.net/jpg/00/87/28/19/240_F_87281963_29bnkFXa6RQnJYWeRfrSpieagNxw1Rru.jpg"
BOT_AVATAR_URL = "https://cdn.pixabay.com/photo/2023/06/23/00/26/ai-generated-8082497_1280.png"

css_layout = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #4c00b0
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 60px;
  max-height: 60x;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="''' + f"{USER_AVATAR_URL}" + '''">
    </div>    
    <div class="message">{msg}</div>
</div>
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="''' + f"{BOT_AVATAR_URL}" + '''">
    </div>
    <div class="message">{msg}</div>
</div>
'''
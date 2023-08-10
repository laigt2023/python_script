import datetime
import websocket
import json

url="ws://192.168.19.240:9090/api/inflet/v1/task/preview?id=a191c139-a214-477f-90f8-de599ad8d1f7&nodeId=Report_v2"

# def on_message(ws, message):
#     print(message)

# ws = websocket.WebSocket()
# ws.on_message = on_message
# ws.connect(url)
# ws.send("Hello, WebSocket!")

import asyncio
import websockets


processTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
frame_rate=0

async def connect_to_websocket():
    global processTime
    global frame_rate
    n = 0
    async with websockets.connect(url) as websocket:
        while True:
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            n+=1
            if nowTime != processTime:
                processTime = nowTime
                frame_rate = n
                n=0

            message = await websocket.recv()
            json_message = json.loads(message)
            print(f"{ nowTime } :  { getFrameRate() }")
            if(json_message.get('details').__len__() > 0):
                targets = json_message.get('details')[0].get('targets')             
                if targets.__len__() > 0:
                    for target in targets:
                        print(f"{ nowTime } : { target.get('label') }")

def getFrameRate():
    global frame_rate
    if frame_rate > 0:
        return f"帧率: { frame_rate }/s"

    return "帧率: 检测中..."     

asyncio.get_event_loop().run_until_complete(connect_to_websocket())            
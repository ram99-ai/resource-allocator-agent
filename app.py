
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from resource_graph import graph  # ✅ this imports your compiled graph
import uuid
app = FastAPI()


#(Optional) Allow local frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    thread_id = "demo"  # you can use UUIDs for unique threads

    state = None

    try:
        while True:
            user_input = await websocket.receive_text()
            if state is None:
                # First user input
                result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": thread_id}})
            else:
                # Continue after HITL interrupt
                result = graph.invoke(Command(resume=user_input), config={"configurable": {"thread_id": thread_id}})

            while "__interrupt__" in result:
                # If graph returns interrupt, ask user
                question = result["__interrupt__"][0].value
                await websocket.send_text(question)
                user_input = await websocket.receive_text()
                result = graph.invoke(Command(resume=user_input), config={"configurable": {"thread_id": thread_id}})

            # Send final output
            await websocket.send_text(str(result["messages"][-1].content))
            state = result

    except WebSocketDisconnect:
        print("WebSocket disconnected")


from fastapi import Body

@app.post("/test")
async def test_query(input_text: str = Body(...)):
    thread_id = str(uuid.uuid4())
    result = graph.invoke(
        {"messages": [HumanMessage(content=input_text)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return result

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     # ✅ Generate a unique thread_id for this connection only
#     thread_id = str(uuid.uuid4())

#     state = None

#     try:
#         while True:
#             user_input = await websocket.receive_text()
#             if state is None:
#                 # First user input
#                 result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": thread_id}})
#             else:
#                 # Continue after HITL interrupt
#                 result = graph.invoke(Command(resume=user_input), config={"configurable": {"thread_id": thread_id}})

#             while "__interrupt__" in result:
#                 # If graph returns interrupt, ask user
#                 question = result["__interrupt__"][0].value
#                 await websocket.send_text(question)
#                 user_input = await websocket.receive_text()
#                 result = graph.invoke(Command(resume=user_input), config={"configurable": {"thread_id": thread_id}})

#             # Send final output
#             await websocket.send_text(str(result))
#             state = result

#     except WebSocketDisconnect:
#         print(f"WebSocket disconnected. Thread: {thread_id}")

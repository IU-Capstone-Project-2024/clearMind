import { NextResponse } from "next/server";


export async function POST(req) {
  let { message, conversationID } = await req.json();
  console.log(message, conversationID)

  const url = "http://20.98.104.160:8000/invoke";
  const cookies = 'user_id=0808080';

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Cookie': cookies
    },
    body: JSON.stringify({
      input: { human_input: message },
      config: {
        configurable: {
          conversation_id: conversationID
        }
      }
    })
  }
  )

  const data = await response.json();
  console.log(data);

  return NextResponse.json({ data }, { status: 200 });

}



// app/api/chat.js
// import { NextResponse } from "next/server";


// export async function POST(req) {
//   let { message, conversationID } = await req.json();
//   console.log(message, conversationID)

//   const url = "http://51.12.240.54:8000/invoke";

//   const response = await fetch(url, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       input: { input: message },
//       config: {},
//       kwargs: {}
//     })
//   }
//   )

//   const data = await response.json();
//   console.log(data);

//   return NextResponse.json({ data }, { status: 200 });

// }

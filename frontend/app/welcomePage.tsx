import { useRouter } from 'next/router';
import { signIn, useSession } from 'next-auth/react';
import { useState } from 'react';

const InputPage = () => {
  const [message, setMessage] = useState('');
  const router = useRouter();
  const { data: session } = useSession();

  const handleSend = async () => {
    if (!session) {
      // Redirect to Google sign-in
      signIn('google', { callbackUrl: `/app/chat?message=${message}` });
    } else {
      // If already signed in, redirect to chat page with message
      router.push(`/app/chat?message=${message}`);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Hello</h1>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message here"
        style={{ marginRight: '10px' }}
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
};

export default InputPage;

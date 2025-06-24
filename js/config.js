// Production URLs - Update these after deploying to Railway
// Replace 'your-app-name' with your actual Railway app name
const BACKEND_URL_PROD = "https://your-app-name.up.railway.app/ask";
const QUIZ_URL_PROD = "https://your-app-name.up.railway.app/quiz";
const QUIZ_EVAL_URL_PROD = "https://your-app-name.up.railway.app/quiz/evaluate";

// Development URLs
const BACKEND_URL_DEV = "http://127.0.0.1:8000/ask";
const QUIZ_URL_DEV = "http://127.0.0.1:8000/quiz";
const QUIZ_EVAL_URL_DEV = "http://127.0.0.1:8000/quiz/evaluate";

// Auto-detect environment
const isDevelopment = window.location.hostname === 'localhost' || 
                     window.location.hostname === '127.0.0.1' ||
                     window.location.protocol === 'file:';

const BACKEND_URL_FINAL = isDevelopment ? BACKEND_URL_DEV : BACKEND_URL_PROD;
const QUIZ_URL_FINAL = isDevelopment ? QUIZ_URL_DEV : QUIZ_URL_PROD;
const QUIZ_EVAL_URL_FINAL = isDevelopment ? QUIZ_EVAL_URL_DEV : QUIZ_EVAL_URL_PROD;

console.log('Environment:', isDevelopment ? 'Development' : 'Production');
console.log('Backend URL:', BACKEND_URL_FINAL);

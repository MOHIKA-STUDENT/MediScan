# MediScan - AI-Powered Medical Report Analyzer 🏥

MediScan is a comprehensive, full-stack application designed to help users and healthcare professionals analyze medical reports (such as Blood Reports and CT Scans) using advanced AI. It provides detailed health summaries, risk assessments, and a historical record of all medical data in a secure and professional dashboard.

## 🚀 Features

- **AI-Driven Report Analysis**: Upload PDFs or images of medical reports and receive a professional, doctor-grade AI summary.
- **Interactive Health Dashboard**: Track health trends, visualize report history, and manage medical documents in one place.
- **Secure Authentication**: Built-in user management with multi-step registration, profile updates, and secure JWT-based login.
- **Cloud-Based Document Storage**: Integrated with Cloudinary for secure and scaleable storage of medical files.
- **Modern UI/UX**: Professional "Dark Pro" aesthetic using Shadcn UI, Framer Motion, and Tailwind CSS.

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 18 (Vite)
- **Styling**: Tailwind CSS, Shadcn UI
- **Animations**: Framer Motion
- **State Management**: TanStack Query (React Query)
- **Routing**: React Router DOM
- **Icons**: Lucide React

### Backend
- **Runtime**: Node.js
- **Framework**: Express.js
- **Database**: MongoDB (Mongoose)
- **Authentication**: JSON Web Tokens (JWT) & Bcrypt
- **File Uploads**: Multer & Cloudinary

## 📦 Installation & Setup

### Prerequisites
- [Node.js](https://nodejs.org/) (v18+ recommended)
- [MongoDB](https://www.mongodb.com/) (Local or Atlas)
- [Cloudinary Account](https://cloudinary.com/) (For file storage)

### Step 1: Clone the Repository
```bash
git clone https://github.com/MOHIKA-STUDENT/MediScan.git
cd MediScan
```

### Step 2: Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `backend/` folder and add the following:
   ```env
   PORT=5001
   MONGO_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```
4. Start the server:
   ```bash
   node server.js
   ```

### Step 3: Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 🛡️ Security & Privacy
MediScan takes medical privacy seriously. All documents are stored securely in the cloud with unique identifiers, and access is strictly controlled via JWT authentication.

## 📄 License
This project is for educational and professional demonstration purposes.

---
Built with ❤️ for better healthcare analytics.

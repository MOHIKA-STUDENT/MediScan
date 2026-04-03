const express = require("express");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const { CloudinaryStorage } = require("multer-storage-cloudinary");
const cloudinary = require("cloudinary").v2;
const Report = require("../models/Report");

const router = express.Router();

// Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});

const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "health_reports",
    allowed_formats: ["jpg", "png", "jpeg", "pdf"],
    resource_type: "auto"
  }
});

const upload = multer({ storage: storage });

// Middleware to verify token
const verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = decoded.id;
    next();
  } catch (err) {
    res.status(401).json({ error: "Invalid token" });
  }
};

// Route to upload a file and save report history
// Expected formData:
// file (the actual file blob)
// reportType (string)
// analysisResult (JSON string)
router.post("/save", verifyToken, upload.single("file"), async (req, res) => {
  try {
    const { reportType, analysisResult } = req.body;
    
    if (!req.file) {
      return res.status(400).json({ error: "No file provided" });
    }

    const newReport = await Report.create({
      userId: req.userId,
      reportType,
      fileUrl: req.file.path, // This is the Cloudinary URL
      analysisResult: typeof analysisResult === 'string' ? JSON.parse(analysisResult) : analysisResult
    });

    res.status(201).json({ message: "Report saved successfully", report: newReport });
  } catch (err) {
    console.error("Error saving report:", err);
    res.status(500).json({ error: "Failed to save report" });
  }
});

// Route to get all reports for the logged in user
router.get("/history", verifyToken, async (req, res) => {
  try {
    const reports = await Report.find({ userId: req.userId }).sort({ createdAt: -1 });
    res.json(reports);
  } catch (err) {
    console.error("Error fetching history:", err);
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

// Route to delete a report
router.delete("/:id", verifyToken, async (req, res) => {
  try {
    const report = await Report.findOneAndDelete({ _id: req.params.id, userId: req.userId });
    if (!report) {
      return res.status(404).json({ error: "Report not found" });
    }
    
    // Attempt to delete from cloudinary if possible (extract public_id)
    try {
      if (report.fileUrl && report.fileUrl.includes('cloudinary.com')) {
        const urlParts = report.fileUrl.split('/');
        const filename = urlParts[urlParts.length - 1];
        const publicId = `health_reports/${filename.split('.')[0]}`;
        await cloudinary.uploader.destroy(publicId);
      }
    } catch (cErr) {
      console.log("Could not delete from cloudinary, but DB record removed:", cErr.message);
    }

    res.json({ message: "Report deleted successfully" });
  } catch (err) {
    console.error("Error deleting report:", err);
    res.status(500).json({ error: "Failed to delete report" });
  }
});

module.exports = router;

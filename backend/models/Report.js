const mongoose = require("mongoose");

const reportSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true
  },
  reportType: {
    type: String,
    enum: ["Blood Report", "CT Scan", "Other"],
    required: true
  },
  fileUrl: {
    type: String, // Cloudinary URL
    required: true
  },
  analysisResult: {
    type: Object, // Store the comprehensive JSON analysis result
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model("Report", reportSchema);

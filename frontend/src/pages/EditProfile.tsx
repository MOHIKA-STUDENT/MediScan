import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { User as UserIcon, Camera } from "lucide-react";

interface User {
  firstName: string;
  lastName: string;
  email: string;
  avatar?: string;
}

const EditProfile = () => {
  const [user, setUser] = useState<User>({ firstName: "", lastName: "", email: "" });
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) return navigate("/login");

      try {
        const res = await fetch("http://localhost:5000/api/auth/profile", {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (res.ok) {
          const data = await res.json();
          setUser(data);
        } else {
          toast.error("Session expired. Please login again.");
          localStorage.removeItem("token");
          navigate("/login");
        }
      } catch (err) {
        toast.error("Failed to fetch profile");
      }
    };

    fetchUser();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    if (!token) return toast.error("Unauthorized");

    try {
      const res = await fetch("http://localhost:5000/api/auth/update", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(user),
      });

      if (res.ok) {
        toast.success("Profile updated successfully");
        navigate("/dashboard");
      } else {
        toast.error("Failed to update profile");
      }
    } catch (err) {
      toast.error("Something went wrong");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md sm:max-w-lg md:max-w-xl bg-white shadow-lg rounded-2xl overflow-hidden">
        
        {/* Card Header with Gradient */}
        <div className="bg-gradient-to-r from-primary to-accent p-6 flex flex-col items-center">
          <div className="relative">
            <div className="bg-white rounded-full p-1">
              <UserIcon className="h-16 w-16 text-primary sm:h-20 sm:w-20 md:h-24 md:w-24" />
            </div>
            <button className="absolute bottom-0 right-0 bg-white p-2 rounded-full shadow hover:bg-gray-100">
              <Camera className="h-4 w-4 sm:h-5 sm:w-5 md:h-6 md:w-6 text-gray-600" />
            </button>
          </div>
          <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-white mt-4 text-center">
            {user.firstName} {user.lastName}
          </h2>
        </div>

        {/* Form */}
        <form className="p-6 space-y-5" onSubmit={handleSubmit}>
          <div className="flex flex-col">
            <label className="mb-2 text-gray-600 font-medium">First Name</label>
            <input
              className="border border-gray-300 px-4 py-3 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none transition w-full"
              value={user.firstName}
              onChange={(e) => setUser({ ...user, firstName: e.target.value })}
              required
            />
          </div>

          <div className="flex flex-col">
            <label className="mb-2 text-gray-600 font-medium">Last Name</label>
            <input
              className="border border-gray-300 px-4 py-3 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none transition w-full"
              value={user.lastName}
              onChange={(e) => setUser({ ...user, lastName: e.target.value })}
              required
            />
          </div>

          <div className="flex flex-col">
            <label className="mb-2 text-gray-600 font-medium">Email</label>
            <input
              type="email"
              className="border border-gray-300 px-4 py-3 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none transition w-full"
              value={user.email}
              onChange={(e) => setUser({ ...user, email: e.target.value })}
              required
            />
          </div>

          <Button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-primary to-accent text-white font-semibold rounded-xl shadow-lg hover:brightness-105 transition-all duration-300"
          >
            Save Changes
          </Button>
        </form>
      </div>
    </div>
  );
};

export default EditProfile;

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import profileImg from "@/assets/profile.png";

interface User {
  firstName: string;
  lastName: string;
  email: string;
}

const Profile = () => {
  const [user, setUser] = useState<User | null>(null);
  const navigate = useNavigate();

  const fetchUser = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      toast.error("Please log in to access your profile");
      navigate("/login");
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/api/auth/profile", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const userData = await res.json();
        setUser(userData);
      } else {
        toast.error("Session expired. Please login again.");
        localStorage.removeItem("token");
        navigate("/login");
      }
    } catch (err) {
      toast.error("Failed to fetch user data");
    }
  };

const handleDelete = () => {
  const toastId = toast(
    <div className="flex flex-col items-center justify-center">
      <p className="text-lg font-medium text-center">
        Are you sure you want to delete your account?
      </p>
      <p className="text-sm text-gray-500 mt-2 text-center">
        This action cannot be undone.
      </p>
      <div className="flex gap-4 mt-4">
        <button
          className="px-6 py-3 bg-red-600 text-white rounded-lg font-semibold text-lg"
          onClick={async () => {
            const token = localStorage.getItem("token");
            if (!token) {
              toast.error("Unauthorized action");
              return;
            }

            try {
              const res = await fetch("http://localhost:5000/api/auth/delete", {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
              });

              if (res.ok) {
                localStorage.removeItem("token");
                toast.success("Account deleted successfully");
                navigate("/signup");
              } else {
                toast.error("Failed to delete account");
              }
            } catch (err) {
              toast.error("Something went wrong while deleting account");
            }

            toast.dismiss(toastId); // dismiss the toast
          }}
        >
          Confirm
        </button>
        <button
          className="px-6 py-3 bg-gray-500 text-white rounded-lg font-semibold text-lg"
          onClick={() => toast.dismiss(toastId)} // dismiss here too
        >
          Cancel
        </button>
      </div>
    </div>,
    {
      duration: 100000, // keep visible until user acts
      style: {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        zIndex: 9999,
        width: "fit-content",
        padding: "1.5rem",
        borderRadius: "0.75rem",
        boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
        backgroundColor: "white",
      },
    }
  );
};



  useEffect(() => {
    fetchUser();
  }, []);

  if (!user) {
    return (
      <p className="text-center text-lg mt-10 text-muted-foreground">Loading...</p>
    );
  }

  return (
    <Card className="p-8 rounded-xl shadow-md bg-muted/50 border border-border max-w-xl mx-auto">
      {/* Default Avatar */}
      <div className="flex flex-col items-center mb-6">
  <img
          src={profileImg} // ✅ fixed
          alt="User Avatar"
          className="w-24 h-24 rounded-full"
          
        />

        <h2 className="text-2xl font-bold font-heading mt-4">
          {user.firstName} {user.lastName}
        </h2>
      </div>

      <div className="space-y-4 text-base text-foreground">
        <div>
          <span className="text-muted-foreground font-medium">Email:</span>{" "}
          {user.email}
        </div>
      </div>

      <div className="flex justify-end gap-4 mt-8">
        <Button onClick={() => navigate("/edit-profile")} className="gap-2">
          <Pencil size={18} />
          Edit
        </Button>
       <Button
  variant="destructive"
  onClick={handleDelete}
  className="gap-2"
>
  <Trash2 size={18} />
  Delete
</Button>

      </div>
    </Card>
  );
};

export default Profile;

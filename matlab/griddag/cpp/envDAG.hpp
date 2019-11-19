
class envDAG {
    private:
    double l, w;

    // This constructor has optional arguments, meaning you can skip them (which will result in them being set to 0).
    public:
    envDAG(const double l = 0, const double w = 0);
   

    double Area(void) const; // the const keyword after the parameter list tells the compiler that this method won't modify the actual object
    double Perim(void) const;
};

